"""Temporal geometry: segment tIoU and union coverage."""
from __future__ import annotations

import numpy as np

__all__ = ["temporal_iou_with_group", "interval_overlap_with_union"]


def temporal_iou_with_group(start, end, ref_starts, ref_ends):
    """tIoU of ``[start, end]`` against each reference segment (0 where the
    union has zero length)."""
    inter = np.minimum(end, ref_ends) - np.maximum(start, ref_starts)
    inter = np.clip(inter, 0.0, None)
    union = (end - start) + (ref_ends - ref_starts) - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def interval_overlap_with_union(seg_start, seg_end, intervals):
    """Length of ``[seg_start, seg_end]`` covered by the union of ``intervals``.

    ``intervals`` is a list of ``(start, end)`` pairs. Overlapping intervals are
    merged before summing, so double-covered regions count once.
    """
    if not intervals:
        return 0.0
    clipped = []
    for start, end in intervals:
        left = max(seg_start, start)
        right = min(seg_end, end)
        if right > left:
            clipped.append((left, right))
    if not clipped:
        return 0.0
    clipped.sort()
    merged = [clipped[0]]
    for start, end in clipped[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return float(sum(end - start for start, end in merged))
