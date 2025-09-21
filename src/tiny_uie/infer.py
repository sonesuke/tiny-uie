"""Inference script for UIE model.

This script performs inference using the trained UIE model for span extraction,
type classification, and relation extraction.
"""

import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from .models.uie import UIE
from .postprocess import enumerate_spans, nms


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
def get_model(model_name: str, ckpt_type: str, ckpt_span: str) -> tuple[UIE, torch.device, str]:
    """Load the UIE model with appropriate checkpoint.

    Args:
        model_name: Name of the pretrained model
        ckpt_type: Path to type classification checkpoint
        ckpt_span: Path to span extraction checkpoint

    Returns:
        Tuple of model, device, and checkpoint path used
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UIE(model_name).to(device)
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
def process_text(
    text: str,
    model: UIE,
    tok: PreTrainedTokenizer | PreTrainedTokenizerFast,
    device: torch.device,
    max_length: int,
) -> tuple:
    """Process input text with the model.

    Args:
        text: Input text to process
        model: UIE model instance
        tok: Tokenizer instance
        device: Device to use for computation
        max_length: Maximum sequence length

    Returns:
        Tuple of model outputs and offsets
    """
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=max_length)
    offsets = enc.pop("offsets_mapping") if "offsets_mapping" in enc else enc.pop("offset_mapping")
    enc = {k: v.to(device) for k, v in enc.items()}

    out, h, _rel_scores = model(**enc)
    start_p = torch.sigmoid(out.start_logits)[0]
    end_p = torch.sigmoid(out.end_logits)[0]
    return start_p, end_p, h, offsets


@torch.no_grad()
def extract_spans(
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
    """Extract and score text spans.

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
    # 候補抽出（TopK & 長さ制限）
    s_idx = torch.topk(start_p, k=min(topk_starts, start_p.numel())).indices.tolist()
    e_idx = torch.topk(end_p, k=min(topk_ends, end_p.numel())).indices.tolist()
    cand = enumerate_spans(s_idx, e_idx, max_span_len)

    # スコア計算（基本は start*end）
    scored = [(st, ed, float(start_p[st] * end_p[ed])) for st, ed in cand if start_p[st] > thr_s and end_p[ed] > thr_e]

    # NMS
    return nms(scored, iou_thr=nms_iou, topk=nms_topk)


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
        # 平均埋め込みで型スコア
        span_vec = torch.nn.functional.normalize(h[0, st : ed + 1].mean(dim=0), dim=-1)
        tscores = (span_vec @ type_emb.T).tolist()
        best = max(range(len(tscores)), key=lambda i: tscores[i])

        # 文字オフセット計算
        # special tokens の offset は (0,0) になることが多いので補正
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
def main() -> None:
    """Main inference function."""
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
    types_path = paths.get("schema_types", "data/schema/types.json")
    ckpt_type = paths.get("ckpt_type", "checkpoints/uie_span_type.pt")
    ckpt_span = paths.get("ckpt_span", "checkpoints/uie_span_only.pt")

    text = sys.argv[1] if len(sys.argv) > 1 else "ABC株式会社は1999年に山田太郎が設立した。"

    tok = AutoTokenizer.from_pretrained(model_name)
    model, device, _ckpt_used = get_model(model_name, ckpt_type, ckpt_span)

    start_p, end_p, h, offsets = process_text(text, model, tok, device, cfg.get("max_length", 1024))

    kept = extract_spans(start_p, end_p, thr_s, thr_e, max_span_len, topk_starts, topk_ends, nms_iou, nms_topk)

    names, type_emb = load_types_and_compute_embeddings(types_path, model, tok, device)

    off = offsets[0].tolist()
    results = compute_type_scores(h, kept, type_emb, names, off, text)

    print(json.dumps({"text": text, "spans": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
