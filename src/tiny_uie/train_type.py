"""Type classification training script for UIE model.

This script trains the type classification component of the UIE model using InfoNCE loss.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional
import yaml
from rich import print
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, get_linear_schedule_with_warmup

from .dataset import UIEDataset
from .models.uie import UIE

Path("checkpoints").mkdir(exist_ok=True)


def info_nce(span_emb: torch.Tensor, type_emb: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """Compute InfoNCE loss for span and type embeddings."""
    logits = (span_emb @ type_emb.T) / tau
    targets = torch.arange(span_emb.size(0), device=span_emb.device)
    return torch.nn.functional.cross_entropy(logits, targets)


@torch.no_grad()
def encode_types(
    model: UIE,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    type_texts: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Encode type texts into embeddings using the model's encoder.

    Args:
        model: UIE model instance
        tokenizer: Tokenizer instance
        type_texts: List of type texts to encode
        device: Device to place tensors on

    Returns:
        Normalized embeddings tensor of shape [N, H]
    """
    toks = tokenizer(type_texts, truncation=True, padding=True, return_tensors="pt", max_length=128)
    toks = {k: v.to(device) for k, v in toks.items()}
    # CLS ベクトルを型表現として使用
    h = model.encoder(**toks).last_hidden_state[:, 0]  # [N,H]
    return torch.nn.functional.normalize(h, dim=-1)


def span_mean(h: torch.Tensor, spans_indices: list[list[tuple[int, int]]]) -> torch.Tensor:
    """Compute mean embeddings for spans.

    Args:
        h: Hidden states tensor of shape [B, T, H]
        spans_indices: List of lists of (start, end) token indices

    Returns:
        Normalized span embeddings tensor of shape [N, H]
    """
    # h: [B,T,H], spans_indices: List[List[(st,ed)]]
    outs = []
    for b, spans in enumerate(spans_indices):
        for st, ed in spans:
            outs.append(h[b, st : ed + 1].mean(dim=0))
    return torch.nn.functional.normalize(torch.stack(outs, dim=0), dim=-1)


def collate(batch: list) -> dict:
    """Collate function for DataLoader.

    Args:
        batch: List of dataset items

    Returns:
        Dictionary with batched tensors and lists
    """
    encs = [b["enc"] for b in batch]
    maxlen = max(len(e["input_ids"]) for e in encs)
    input_ids, attn = [], []
    for e in encs:
        pad = maxlen - len(e["input_ids"])
        input_ids.append(e["input_ids"] + [0] * pad)
        attn.append(e["attention_mask"] + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attn),
        "spans": [b["spans"] for b in batch],
        "spans_tok": [b["spans_tok"] for b in batch],
    }


def main() -> None:
    """Main training function."""
    with Path("config.yaml").open() as f:
        cfg = yaml.safe_load(f)
    model_name = cfg["model_name"]
    paths = cfg.get("paths", {})
    train_path = paths.get("train", "data/train.jsonl")
    types_path = paths.get("schema_types", "data/schema/types.json")
    ckpt_span = paths.get("ckpt_span", "checkpoints/uie_span_only.pt")
    ckpt_out = paths.get("ckpt_type", "checkpoints/uie_span_type.pt")

    ds = UIEDataset(train_path, model_name, cfg.get("max_length", 1024))
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UIE(model_name).to(device)
    if Path(ckpt_span).exists():
        model.load_state_dict(torch.load(ckpt_span, map_location=device), strict=False)
        print(f"[info] loaded span checkpoint: {ckpt_span}")
    else:
        print(f"[warn] span checkpoint not found: {ckpt_span} (cold start)")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    total_steps = len(dl) * cfg["train"]["epochs"]
    sch = get_linear_schedule_with_warmup(opt, int(total_steps * cfg["train"]["warmup_ratio"]), total_steps)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    with Path(types_path).open(encoding="utf-8") as f:
        types = json.load(f)
    type_names = list(types.keys())
    type_texts = [types[t]["name"] + "：" + types[t]["definition"] for t in type_names]
    type_emb = encode_types(model, tokenizer, type_texts, device)  # [K,H]

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for step, batch_item in enumerate(dl):
            batch_data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch_item.items()}
            out, h, _rel_scores = model(batch_data["input_ids"], batch_data["attention_mask"])  # h: [B,T,H]
            # gold span のトークン範囲（学習データの spans_tok を使用）
            spans_indices = [[(st.start_tok, st.end_tok_inclusive) for st in arr] for arr in batch_data["spans_tok"]]
            # gold type の index 列
            gold_idx = []
            for spans in batch_data["spans"]:
                for s in spans:
                    gold_idx.append(type_names.index(s["type"]))
            # span 埋め込み
            span_emb = span_mean(h, spans_indices)  # [N,H]
            pos_type = type_emb[torch.tensor(gold_idx, device=device)]  # [N,H]
            loss = info_nce(span_emb, pos_type, tau=0.07)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            opt.zero_grad()
            if step % 50 == 0:
                print(f"[epoch {epoch}] step {step} loss {loss.item():.4f}")

    torch.save(model.state_dict(), ckpt_out)
    print("saved to", ckpt_out)


if __name__ == "__main__":
    main()
