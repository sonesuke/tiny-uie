"""Relation inference script for UIE model.

This script performs relation extraction inference using the trained UIE model.
"""

import json
import sys
import unicodedata
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from .models.uie import UIE
from .postprocess import enumerate_spans, nms

# Added: Utility to filter punctuation, symbols, and function words
STOPWORDS = {"の", "は", "である", "。", "、"}


def is_punct_or_symbol(s: str) -> bool:
    """Check if a string consists only of punctuation or symbols.

    Args:
        s: Input string to check

    Returns:
        True if all characters are punctuation/symbols or whitespace, False otherwise
    """
    return all(unicodedata.category(ch).startswith(("P", "S")) or ch.isspace() for ch in s)


def clean_and_score_spans(
    raw_spans: list, h: torch.Tensor, type_emb: torch.Tensor, type_names: list, type_min: float = 0.6
) -> list:
    """Process raw spans and assign types with filtering.

    Args:
        raw_spans: List of spans in format [(st, ed, base_score)]
        h: Hidden states
        type_emb: Type embeddings
        type_names: List of type names
        type_min: Minimum type score threshold

    Returns:
        List of processed spans with type assignments
    """
    kept = []
    for st, ed, base_score in raw_spans:
        # Safely get character range
        # offsets are extracted from enc's offset_mapping in main()
        # → If we need to increase function parameters for safe access,
        #    adjust char range and text at the caller side - OK here
        # Only calculate span_vec for type scoring
        span_vec = torch.nn.functional.normalize(h[0, st : ed + 1].mean(dim=0), dim=-1)
        scores = (span_vec @ type_emb.T).tolist()
        best = max(range(len(scores)), key=lambda i: scores[i])
        kept.append((st, ed, base_score, type_names[best], float(scores[best])))
    # Filter
    out = []
    for st, ed, base, ty, tscore in kept:
        # Check string length, punctuation/symbols, and stopwords at the caller side using text
        out.append({"st": st, "ed": ed, "score_span": base, "type": ty, "score_type": tscore})
    # Apply minimum threshold
    return [s for s in out if s["score_type"] >= type_min]


def finalize_spans(spans_struct: list, text: str, offsets: list, min_char_len: int = 2) -> list:
    """Attach char ranges and text, then filter out punctuation/function words."""
    off = offsets[0].tolist()
    finalized = []
    for s in spans_struct:
        st, ed = s["token_start"], s["token_end"]
        char_s = None
        char_e = None
        for i in range(st, ed + 1):
            cs, ce = off[i]
            if cs == 0 and ce == 0:  # special token
                continue
            char_s = cs if char_s is None else min(char_s, cs)
            char_e = ce if char_e is None else max(char_e, ce)
        if char_s is None or char_e is None:
            continue
        frag = text[char_s:char_e]
        if len(frag) < min_char_len:  # Filter short spans
            continue
        if frag in STOPWORDS or is_punct_or_symbol(frag):
            continue
        finalized.append({**s, "char_start": int(char_s), "char_end": int(char_e), "text": frag})
    return finalized


def valid_pair_by_constraints(h_span: dict, t_span: dict, rel_name: str, constraints: dict) -> bool:
    """Check if a pair of spans is valid according to relation constraints.

    Args:
        h_span: Head span dictionary
        t_span: Tail span dictionary
        rel_name: Relation name
        constraints: Dictionary of relation constraints

    Returns:
        True if the pair is valid according to constraints, False otherwise
    """
    if rel_name not in constraints:
        return True
    head_ok, tail_ok = constraints[rel_name]
    if head_ok and h_span["type"] not in head_ok:
        return False
    return not (tail_ok and t_span["type"] not in tail_ok)


# Constants for thresholds and limits
MIN_SPAN_COUNT = 2
MAX_PAIR_DISTANCE = 30
MAX_SPAN_LENGTH = 12
SPAN_THRESHOLD = 0.5


@torch.no_grad()
def encode_types(
    model: UIE, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, types: dict, device: torch.device
) -> tuple[list, torch.Tensor]:
    """Encode type texts into embeddings using the model's encoder.

    Args:
        model: UIE model instance
        tokenizer: AutoTokenizer instance
        types: Dictionary of type definitions
        device: Device to place tensors on

    Returns:
        Tuple of type names and embeddings tensor
    """
    names = list(types.keys())
    texts = [types[t]["name"] + "：" + types[t]["definition"] for t in names]
    toks = tokenizer(texts, truncation=True, padding=True, return_tensors="pt", max_length=128)
    toks = {k: v.to(device) for k, v in toks.items()}
    h = model.encoder(**toks).last_hidden_state[:, 0]
    emb = torch.nn.functional.normalize(h, dim=-1)
    return names, emb


@torch.no_grad()
def load_config() -> dict:
    """Load configuration from config.yaml file."""
    with Path("config.yaml").open() as f:
        return yaml.safe_load(f)


@torch.no_grad()
def get_model(model_name: str, ckpt_type: str, ckpt_span: str, rels_path: str) -> tuple[UIE, torch.device, str]:
    """Load the UIE model with appropriate checkpoint.

    Args:
        model_name: Name of the pretrained model
        ckpt_type: Path to type classification checkpoint
        ckpt_span: Path to span extraction checkpoint
        rels_path: Path to relations schema

    Returns:
        Tuple of model, device, and checkpoint path used
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load relations schema to get n_rel
    with Path(rels_path).open(encoding="utf-8") as f:
        rels_schema = json.load(f)
    rel_names = list(rels_schema.keys())

    model = UIE(model_name, n_rel=len(rel_names)).to(device)
    ckpt_used = "(none)"

    if Path(ckpt_type).exists():
        model.load_state_dict(torch.load(ckpt_type, map_location=device), strict=False)
        ckpt_used = ckpt_type
    elif Path(ckpt_span).exists():
        model.load_state_dict(torch.load(ckpt_span, map_location=device), strict=False)
        ckpt_used = ckpt_span

    print(f"[info] checkpoint: {ckpt_used}")
    return model, device, ckpt_used


@torch.no_grad()
def load_types_and_compute_embeddings(
    types_path: str, model: UIE, tok: PreTrainedTokenizer | PreTrainedTokenizerFast, device: torch.device
) -> tuple[list, torch.Tensor]:
    """Load type definitions and compute their embeddings.

    Args:
        types_path: Path to types.json file
        model: UIE model instance
        tok: Tokenizer instance
        device: Device to use for computation

    Returns:
        Tuple of type names and embeddings
    """
    with Path(types_path).open(encoding="utf-8") as f:
        types = json.load(f)
    return encode_types(model, tok, types, device)


@torch.no_grad()
def extract_spans_nms(
    start_p: torch.Tensor,
    end_p: torch.Tensor,
    thr_s: float,
    thr_e: float,
    max_span_len: int,
    topk_starts: int,
    topk_ends: int,
    nms_iou: float,
    nms_topk: int,
) -> list:
    """Extract and score text spans using NMS.

    Args:
        start_p: Start position probabilities
        end_p: End position probabilities
        thr_s: Start threshold
        thr_e: End threshold
        max_span_len: Maximum span length
        topk_starts: Number of top start positions
        topk_ends: Number of top end positions
        nms_iou: NMS IoU threshold
        nms_topk: NMS top-k value

    Returns:
        List of kept spans
    """
    # Candidate extraction (TopK & length limit)
    s_idx = torch.topk(start_p, k=min(topk_starts, start_p.numel())).indices.tolist()
    e_idx = torch.topk(end_p, k=min(topk_ends, end_p.numel())).indices.tolist()
    cand = enumerate_spans(s_idx, e_idx, max_span_len)

    # Score calculation (start*end)
    scored = [(st, ed, float(start_p[st] * end_p[ed])) for st, ed in cand if start_p[st] > thr_s and end_p[ed] > thr_e]

    # NMS
    return nms(scored, iou_thr=nms_iou, topk=nms_topk)


@torch.no_grad()
def compute_type_scores(h: torch.Tensor, kept: list, type_emb: torch.Tensor, names: list, off: list, text: str) -> list:
    """Compute type scores for extracted spans.

    Args:
        h: Hidden states from model
        kept: List of kept spans
        type_emb: Type embeddings
        names: Type names
        off: Offset mappings
        text: Input text

    Returns:
        List of results with type scores
    """
    results = []
    for st, ed, base_score in kept:
        # Average embedding for type scoring
        span_vec = torch.nn.functional.normalize(h[0, st : ed + 1].mean(dim=0), dim=-1)
        tscores = (span_vec @ type_emb.T).tolist()
        best = max(range(len(tscores)), key=lambda i: tscores[i])

        # Character offset calculation
        # special tokens' offset is often (0,0), so we need to correct for that
        char_s = None
        char_e = None
        for i in range(st, ed + 1):
            s, e = off[i]
            if s == 0 and e == 0:
                continue
            char_s = s if char_s is None else min(char_s, s)
            char_e = e if char_e is None else max(char_e, e)

        span_text = text[char_s:char_e] if (char_s is not None and char_e is not None) else ""
        results.append(
            {
                "text": span_text,
                "char_start": char_s,
                "char_end": char_e,
                "token_start": int(st),
                "token_end": int(ed),
                "type": names[best],
                "score_base": base_score,
                "score_type": float(tscores[best]),
            }
        )
    return results


@torch.no_grad()
def infer_relations(
    model: UIE, spans: list, hidden_states: torch.Tensor, rels_schema: dict, inference_config: dict
) -> list:
    """Extract relations between spans.

    Args:
        model: Trained UIE model
        spans: List of extracted spans with their types and scores
        hidden_states: Hidden states from the model
        rels_schema: Relation schema dictionary
        inference_config: Inference configuration dictionary

    Returns:
        List of extracted relations
    """
    if len(spans) < MIN_SPAN_COUNT:
        return []

    rel_names = list(rels_schema.keys())
    id_to_rel = {i + 1: name for i, name in enumerate(rel_names)}

    # Load constraints from relation schema
    constraints = {}
    for rel_name, rel_info in rels_schema.items():
        if "constraints" in rel_info:
            constraints[rel_name] = (rel_info["constraints"].get("head", []), rel_info["constraints"].get("tail", []))

    rels = []

    def span_vec(ts: int, te: int) -> torch.Tensor:
        return hidden_states[0, ts : te + 1].mean(dim=0)

    # For each pair of spans
    for i, h_span in enumerate(spans):
        for j, t_span in enumerate(spans):
            if i == j:
                continue

            # Distance constraint: filter pairs that are too far apart
            h_pos = h_span["token_start"]
            t_pos = t_span["token_start"]
            h_end_pos = h_span["token_end"]
            t_end_pos = t_span["token_end"]

            if abs(t_pos - h_end_pos) > MAX_PAIR_DISTANCE and abs(h_pos - t_end_pos) > MAX_PAIR_DISTANCE:
                continue

            # Get head and tail embeddings by averaging token embeddings
            hv = span_vec(h_span["token_start"], h_span["token_end"])
            tv = span_vec(t_span["token_start"], t_span["token_end"])

            # Compute relation logits
            if model.rel is not None:
                logit = model.rel(hv[None, :], tv[None, :])  # [1, n_rel]
            else:
                continue
            prob = torch.softmax(logit, dim=-1)[0]

            # Get the most likely relation
            rid = int(prob.argmax())
            if rid == 0:
                continue

            rel_name = id_to_rel.get(rid, rid)

            # Check constraints if they exist
            if not valid_pair_by_constraints(h_span, t_span, rel_name, constraints):
                continue

            # Threshold for relation extraction
            if float(prob[rid]) < inference_config.get("relation_threshold", 0.5):
                continue

            rels.append(
                {
                    "head": i,
                    "tail": j,
                    "type": rel_name,
                    "score": float(prob[rid]),
                }
            )

    return rels


@torch.no_grad()
def main() -> None:
    """Main relation inference function."""
    cfg = load_config()
    model_name = cfg["model_name"]
    infer = cfg.get("inference", {})
    thr_s = infer.get("thr_start", 0.5)
    thr_e = infer.get("thr_end", 0.5)
    max_span_len = infer.get("max_span_len", 12)
    topk_starts = infer.get("topk_starts", 20)
    topk_ends = infer.get("topk_ends", 20)
    nms_iou = infer.get("nms_iou", 0.5)
    nms_topk = infer.get("nms_topk", 50)

    paths = cfg.get("paths", {})
    rels_path = paths.get("schema_relations", "data/schema/relations.json")
    types_path = paths.get("schema_types", "data/schema/types.json")
    ckpt_rel = paths.get("ckpt_rel", "checkpoints/uie_span_rel.pt")
    ckpt_type = paths.get("ckpt_type", "checkpoints/uie_span_type.pt")
    ckpt_span = paths.get("ckpt_span", "checkpoints/uie_span_only.pt")

    # Load relations schema
    with Path(rels_path).open(encoding="utf-8") as f:
        rels_schema = json.load(f)

    text = sys.argv[1] if len(sys.argv) > 1 else "ABC株式会社は1999年に山田太郎が設立した。"

    # Load model
    model, device, _ckpt_used = get_model(model_name, ckpt_type, ckpt_span, rels_path)

    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=1024)
    offsets = enc.pop("offsets_mapping") if "offsets_mapping" in enc else enc.pop("offset_mapping")
    enc = {k: v.to(device) for k, v in enc.items()}

    # Load relation checkpoint if it exists
    if Path(ckpt_rel).exists():
        model.load_state_dict(torch.load(ckpt_rel, map_location=device))
        print(f"[info] loaded relation checkpoint: {ckpt_rel}")

    # Run model inference
    out, h, _rel_scores = model(**enc)

    # Extract spans with NMS
    start_p = torch.sigmoid(out.start_logits)[0]
    end_p = torch.sigmoid(out.end_logits)[0]

    kept = extract_spans_nms(start_p, end_p, thr_s, thr_e, max_span_len, topk_starts, topk_ends, nms_iou, nms_topk)

    # Load type embeddings and compute type scores for spans
    names, type_emb = load_types_and_compute_embeddings(types_path, model, tokenizer, device)
    off = offsets[0].tolist()
    spans = compute_type_scores(h, kept, type_emb, names, off, text)

    # Filter spans by type score and finalize
    spans = [s for s in spans if s["score_type"] >= infer.get("type_min", 0.6)]
    spans = finalize_spans(spans, text, offsets, infer.get("min_char_len", 2))

    # Extract relations
    relations = infer_relations(model, spans, h, rels_schema, infer)

    # Output results
    result = {"text": text, "spans": spans, "relations": relations}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
