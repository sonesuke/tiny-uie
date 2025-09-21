"""Post-processing functions for UIE model inference.

This module contains functions for non-maximum suppression and span enumeration.
"""


def iou_tok(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Calculate Intersection over Union (IoU) for two token spans.

    Args:
        a: First span as (start, end) tuple with inclusive indices
        b: Second span as (start, end) tuple with inclusive indices

    Returns:
        IoU score between the two spans
    """
    # inclusive indices [st, ed]
    st = max(a[0], b[0])
    ed = min(a[1], b[1])
    inter = max(0, ed - st + 1)
    len_a = a[1] - a[0] + 1
    len_b = b[1] - b[0] + 1
    union = len_a + len_b - inter
    return 0.0 if union == 0 else inter / union


def nms(spans: list[tuple[int, int, float]], iou_thr: float, topk: int) -> list[tuple[int, int, float]]:
    """Apply Non-Maximum Suppression (NMS) to filter overlapping spans.

    Args:
        spans: List of spans as (start, end, score) tuples
        iou_thr: IoU threshold for suppression
        topk: Maximum number of spans to keep

    Returns:
        Filtered list of spans
    """
    # spans: (st, ed, score)
    spans = sorted(spans, key=lambda x: x[2], reverse=True)
    kept = []
    for s in spans:
        if all(iou_tok((s[0], s[1]), (k[0], k[1])) < iou_thr for k in kept):
            kept.append(s)
        if len(kept) >= topk:
            break
    return kept


def enumerate_spans(start_idx: list[int], end_idx: list[int], max_len: int) -> list[tuple[int, int]]:
    """Enumerate all possible span combinations from start and end indices.

    Args:
        start_idx: List of start indices
        end_idx: List of end indices
        max_len: Maximum span length

    Returns:
        List of valid span tuples (start, end)
    """
    out = []
    for st in start_idx:
        for ed in end_idx:
            if ed >= st and (ed - st + 1) <= max_len:
                out.append((st, ed))
    return out
