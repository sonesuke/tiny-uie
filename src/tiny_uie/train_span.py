"""Span extraction training script for UIE model.

This script trains the span extraction component of the UIE model using focal loss.
"""

from pathlib import Path

import torch
import torch.nn.functional
import yaml
from rich import print
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .dataset import UIEDataset
from .models.uie import UIE

# Constants
THRESHOLD = 0.5

Path("checkpoints").mkdir(exist_ok=True)


def focal_bce(logits: Tensor, targets: Tensor, gamma: float = 2.0) -> Tensor:
    """Compute focal binary cross entropy loss."""
    p = torch.sigmoid(logits)
    ce = torch.nn.functional.binary_cross_entropy(p, targets.float(), reduction="none")
    pt = torch.where(targets == 1, p, 1 - p)
    return ((1 - pt) ** gamma * ce).mean()


def make_targets(lengths: list[int], spans_tok_batch: list, device: torch.device) -> tuple[Tensor, Tensor]:
    """Create target tensors for span start and end positions.

    Args:
        lengths: List of sequence lengths for each batch item
        spans_tok_batch: List of token-level spans for each batch item
        device: Device to place the tensors on

    Returns:
        Tuple of start and end target tensors
    """
    batch_size = len(spans_tok_batch)
    max_length = max(lengths)
    start = torch.zeros(batch_size, max_length, device=device)
    end = torch.zeros(batch_size, max_length, device=device)
    for batch_idx in range(batch_size):
        for span_tok in spans_tok_batch[batch_idx]:
            start_tok, end_tok = span_tok.start_tok, span_tok.end_tok_inclusive
            if start_tok < lengths[batch_idx]:
                start[batch_idx, start_tok] = 1
            if end_tok < lengths[batch_idx]:
                end[batch_idx, end_tok] = 1
    return start, end


def bin_prf1(pred: torch.Tensor, gold: torch.Tensor) -> tuple[float, float, float]:
    """Calculate precision, recall, and F1 score for binary classification.

    Args:
        pred: Predicted boolean tensor of shape [B, T]
        gold: Ground truth boolean tensor of shape [B, T]

    Returns:
        Tuple of (precision, recall, f1_score)
    """
    # pred, gold: Bool tensor [B,T]
    tp = (pred & gold).sum().item()
    p = tp / max(pred.sum().item(), 1)
    r = tp / max(gold.sum().item(), 1)
    f = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return p, r, f


def evaluate(dev_dl: DataLoader, model: UIE, device: torch.device) -> dict:
    """Evaluate the model on development data.

    Args:
        dev_dl: Development data loader
        model: UIE model instance
        device: Device to run evaluation on

    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()

    tot = 0
    ps = pr = fs = 0.0
    pe = re = fe = 0.0
    with torch.no_grad():
        for batch_item in dev_dl:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch_item.items()}
            (out, _, _) = model(batch["input_ids"], batch["attention_mask"])
            start_t, end_t = make_targets(batch["lengths"], batch["spans_tok"], device)
            start_p = torch.sigmoid(out.start_logits) > THRESHOLD
            end_p = torch.sigmoid(out.end_logits) > THRESHOLD
            ps_, rs_, fs_ = bin_prf1(start_p, (start_t > THRESHOLD))
            pe_, re_, fe_ = bin_prf1(end_p, (end_t > THRESHOLD))
            ps += ps_
            pr += rs_
            fs += fs_
            pe += pe_
            re += re_
            fe += fe_
            tot += 1
    if tot == 0:
        return {}
    return {
        "start_P": ps / tot,
        "start_R": pr / tot,
        "start_F1": fs / tot,
        "end_P": pe / tot,
        "end_R": re / tot,
        "end_F1": fe / tot,
    }


def collate(batch: list) -> dict:
    """Collate function for DataLoader.

    Args:
        batch: List of dataset items

    Returns:
        Dictionary with batched tensors and lists
    """
    encs = [b["enc"] for b in batch]
    maxlen = max(len(e["input_ids"]) for e in encs)
    input_ids, attn, lengths = [], [], []
    for e in encs:
        length = len(e["input_ids"])
        lengths.append(length)
        pad = maxlen - length
        input_ids.append(e["input_ids"] + [0] * pad)
        attn.append(e["attention_mask"] + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attn),
        "lengths": lengths,
        "spans_tok": [b["spans_tok"] for b in batch],
        "gold_spans": [b["spans"] for b in batch],
    }


def main() -> None:
    """Main training function."""
    with Path("config.yaml").open() as f:
        cfg = yaml.safe_load(f)
    train = UIEDataset(cfg["paths"]["train"], cfg["model_name"], cfg["max_length"])
    dev = UIEDataset(cfg["paths"]["dev"], cfg["model_name"], cfg["max_length"])
    dl = DataLoader(train, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UIE(cfg["model_name"]).to(device)

    # Load checkpoint if it exists
    ckpt_span_path = cfg["paths"]["ckpt_span"]
    if Path(ckpt_span_path).exists():
        model.load_state_dict(torch.load(ckpt_span_path, map_location=device))
        print(f"[epoch 0/{cfg['train']['epochs']}] loaded checkpoint: {ckpt_span_path}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    total_steps = len(dl) * cfg["train"]["epochs"]
    sch = get_linear_schedule_with_warmup(opt, int(total_steps * cfg["train"]["warmup_ratio"]), total_steps)

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        running = 0.0
        pos_rate_acc = 0.0
        for step, batch_item in enumerate(dl):
            batch_data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch_item.items()}
            out, _h, _rel_scores = model(batch_data["input_ids"], batch_data["attention_mask"])
            start_t, end_t = make_targets(batch_data["lengths"], batch_data["spans_tok"], device)
            loss = focal_bce(out.start_logits, start_t) + focal_bce(out.end_logits, end_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            opt.zero_grad()

            running += loss.item()
            # 正例率のざっくり可視化（どのくらい1があるか）
            pos_rate = (start_t.mean().item() + end_t.mean().item()) / 2.0
            pos_rate_acc += pos_rate

            if (step + 1) % 10 == 0 or (step + 1) == len(dl):
                avg = running / 10 if (step + 1) % 10 == 0 else running / ((step + 1) % 10)
                prate = pos_rate_acc / 10 if (step + 1) % 10 == 0 else pos_rate_acc / ((step + 1) % 10)
                print(
                    f"[epoch {epoch + 1}/{cfg['train']['epochs']}] step {step + 1}/{len(dl)}  "
                    f"loss(avg10) {avg:.4f}  pos_rate~{prate:.4f}"
                )
                running = 0.0
                pos_rate_acc = 0.0

        torch.save(model.state_dict(), cfg["paths"]["ckpt_span"])

        # 各エポック終わりで dev をざっくり評価
        dev_dl = DataLoader(dev, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate)
        metrics = evaluate(dev_dl, model, device)
        if metrics:
            print(
                f"[epoch {epoch + 1}/{cfg['train']['epochs']}] dev: "
                f"startF1 {metrics['start_F1']:.3f} (P {metrics['start_P']:.3f}/R {metrics['start_R']:.3f}) | "
                f"endF1 {metrics['end_F1']:.3f} (P {metrics['end_P']:.3f}/R {metrics['end_R']:.3f})"
            )
    print(f"[epoch {cfg['train']['epochs']}/{cfg['train']['epochs']}] saved to", cfg["paths"]["ckpt_span"])


if __name__ == "__main__":
    main()
