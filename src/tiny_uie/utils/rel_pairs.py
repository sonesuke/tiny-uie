"""Relation pair utilities for UIE model.

This module provides functions for building span pairs and generating relation labels.
"""

import random

import torch


def gold_pairs(
    spans_tok: list, relations: list, rel_name_to_id: dict
) -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    """Build gold head-tail pairs with relation IDs.

    Args:
        spans_tok: List of token spans
        relations: List of relation dictionaries with head, tail, type
        rel_name_to_id: Dictionary mapping relation names to IDs

    Returns:
        List of triplets ((head_st, head_ed), (tail_st, tail_ed), rel_id)
    """
    pairs = []  # [((h_st,h_ed),(t_st,t_ed), label_id)]
    for r in relations:
        hi, ti = r["head"], r["tail"]
        h = spans_tok[hi]
        t = spans_tok[ti]
        if r["type"] in rel_name_to_id:
            pairs.append(
                ((h.start_tok, h.end_tok_inclusive), (t.start_tok, t.end_tok_inclusive), rel_name_to_id[r["type"]])
            )
    return pairs


def all_pairs(spans_tok: list) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Generate all possible span pairs (excluding self pairs).

    Args:
        spans_tok: List of token spans

    Returns:
        List of pairs ((head_st, head_ed), (tail_st, tail_ed))
    """
    idx = list(range(len(spans_tok)))
    out = []
    for i in idx:
        for j in idx:
            if i == j:
                continue
            h, t = spans_tok[i], spans_tok[j]
            out.append(((h.start_tok, h.end_tok_inclusive), (t.start_tok, t.end_tok_inclusive)))
    return out


def make_rel_labels(pairs_all: list, gold_triplets: list) -> torch.Tensor:
    """Create relation labels for all pairs.

    Args:
        pairs_all: List of all span pairs
        gold_triplets: List of gold triplets with relation IDs

    Returns:
        Label tensor with relation IDs
    """
    # pairs_all: [((h_st,h_ed),(t_st,t_ed))]
    # gold_triplets: [((h_st,h_ed),(t_st,h_ed), rel_id>=1)]
    label = torch.zeros(len(pairs_all), dtype=torch.long)
    gold_map = {(hp, tp): rid for (hp, tp, rid) in gold_triplets}
    for k, (hp, tp) in enumerate(pairs_all):
        label[k] = gold_map.get((hp, tp), 0)
    return label


def sample_pairs(pairs_all: list, gold_set: set, max_neg: int = 32) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Sample negative pairs to limit total pairs.

    Args:
        pairs_all: List of all span pairs
        gold_set: Set of gold pairs
        max_neg: Maximum number of negative samples

    Returns:
        List of sampled pairs (gold + limited negatives)
    """
    neg = [p for p in pairs_all if p not in gold_set]
    random.shuffle(neg)
    return list(gold_set) + neg[:max_neg]
