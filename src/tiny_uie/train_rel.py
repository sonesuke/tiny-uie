"""Relation extraction training script for UIE model.

This script trains the relation extraction component of the UIE model using cross entropy loss.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional
import yaml
from rich import print
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .dataset import UIEDataset
from .models.uie import UIE
from .utils.rel_pairs import all_pairs, gold_pairs, make_rel_labels, sample_pairs

# Constants
MIN_SPAN_COUNT = 2

Path("checkpoints").mkdir(exist_ok=True)


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
        "relations": [b["relations"] for b in batch],
        "spans_tok": [b["spans_tok"] for b in batch],
    }


def process_batch(batch_data: dict, model: UIE, h: torch.Tensor, rel_to_id: dict) -> float:
    """Process a batch of data and compute loss.

    Args:
        batch_data: Batch data dictionary
        model: UIE model instance
        h: Hidden states
        rel_to_id: Relation name to ID mapping

    Returns:
        Computed batch loss
    """
    batch_loss = 0.0
    batch_size = batch_data["input_ids"].size(0)

    for b in range(batch_size):
        spans_tok_b = batch_data["spans_tok"][b]
        rels_b = batch_data["relations"][b]

        if len(spans_tok_b) < MIN_SPAN_COUNT:  # Need at least 2 spans to form pairs
            continue

        # Generate all possible pairs
        pairs_all = all_pairs(spans_tok_b)
        if not pairs_all:
            continue

        # Get gold triplets
        triplets = gold_pairs(spans_tok_b, rels_b, rel_to_id)
        gold_set = {(hp, tp) for (hp, tp, _) in triplets}

        # Sample pairs to limit negative examples
        sel_pairs = sample_pairs(pairs_all, gold_set, max_neg=64)

        # Create embeddings for selected pairs
        head_vec, tail_vec = [], []
        for hp, tp in sel_pairs:
            h_st, h_ed = hp
            t_st, t_ed = tp
            head_vec.append(h[b, h_st : h_ed + 1].mean(dim=0))
            tail_vec.append(h[b, t_st : t_ed + 1].mean(dim=0))

        if head_vec and tail_vec:
            head_vec = torch.stack(head_vec)
            tail_vec = torch.stack(tail_vec)

            # Create labels and compute relation scores
            labels = make_rel_labels(sel_pairs, triplets)
            if model.rel is not None:
                rel_logits = model.rel(head_vec, tail_vec)  # [M, n_rel]
                # Compute loss
                batch_loss += torch.nn.functional.cross_entropy(rel_logits, labels.to(rel_logits.device)).item()
            else:
                continue
    return batch_loss


def main() -> None:
    """Main training function."""
    with Path("config.yaml").open() as f:
        cfg = yaml.safe_load(f)
    model_name = cfg["model_name"]
    paths = cfg.get("paths", {})
    train_path = paths.get("train", "data/train.jsonl")
    rels_path = paths.get("schema_relations", "data/schema/relations.json")
    ckpt_in = paths.get("ckpt_type", "checkpoints/uie_span_type.pt")
    ckpt_out = paths.get("ckpt_rel", "checkpoints/uie_span_rel.pt")

    ds = UIEDataset(train_path, model_name, cfg.get("max_length", 1024))
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate)

    with Path(rels_path).open(encoding="utf-8") as f:
        rels = json.load(f)
    rel_names = list(rels.keys())

    rel_to_id = {name: i for i, name in enumerate(rel_names)}
    # id_to_rel = {v: k for k, v in rel_to_id.items()}  # Commented out unused variable

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UIE(model_name, n_rel=len(rel_names)).to(device)
    if Path(ckpt_in).exists():
        model.load_state_dict(torch.load(ckpt_in, map_location=device), strict=False)
        print(f"[info] loaded type checkpoint: {ckpt_in}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    total_steps = len(dl) * cfg["train"]["epochs"]
    sch = get_linear_schedule_with_warmup(opt, int(total_steps * cfg["train"]["warmup_ratio"]), total_steps)

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for step, batch_item in enumerate(dl):
            batch_data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch_item.items()}

            # Get model outputs
            (_out, h, _rel_scores) = model(batch_data["input_ids"], batch_data["attention_mask"])

            # Process batch and compute loss
            batch_loss = process_batch(batch_data, model, h, rel_to_id)

            batch_size = batch_data["input_ids"].size(0)
            if batch_size > 0:
                loss = batch_loss / batch_size
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
