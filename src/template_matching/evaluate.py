"""Evaluation of detections against known ground truth.

A detection counts as a true positive when its IoU with an unmatched ground
truth box reaches ``iou_threshold`` (0.5 by convention, the PASCAL VOC
criterion — Everingham et al. 2010).  Matching is greedy by score, best
detection first, so each ground truth box can be claimed at most once.
"""

from __future__ import annotations

from .matching import MatchResult, iou


def evaluate_detections(
    detections: list[MatchResult],
    ground_truth: list[dict],
    iou_threshold: float = 0.5,
) -> dict:
    """Score detections against ground truth; return precision/recall/F1.

    ``ground_truth`` entries are dicts with ``x``, ``y``, ``w``, ``h`` (extra
    keys such as ``shape`` are ignored).  The returned dict also carries the
    per-detection verdicts (``"tp"`` / ``"fp"``) and the list of missed
    ground-truth boxes, so callers can report *which* instances failed.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    gt_boxes = [(g["x"], g["y"], g["w"], g["h"]) for g in ground_truth]
    claimed = [False] * len(gt_boxes)
    verdicts: list[dict] = []
    tp = 0
    # Greedy best-first matching; "best" means highest similarity or lowest
    # distance, depending on the method of the detections.
    from .metrics import is_similarity

    similarity = is_similarity(detections[0].method) if detections else True
    for det in sorted(detections, key=lambda d: d.score, reverse=similarity):
        best_iou, best_idx = 0.0, -1
        for idx, box in enumerate(gt_boxes):
            if claimed[idx]:
                continue
            value = iou(det.box(), box)
            if value > best_iou:
                best_iou, best_idx = value, idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            claimed[best_idx] = True
            tp += 1
            verdicts.append({"detection": det.to_dict(), "verdict": "tp", "iou": round(best_iou, 4)})
        else:
            verdicts.append({"detection": det.to_dict(), "verdict": "fp", "iou": round(best_iou, 4)})
    fp = len(detections) - tp
    fn = len(gt_boxes) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "iou_threshold": iou_threshold,
        "verdicts": verdicts,
        "missed": [ground_truth[i] for i, c in enumerate(claimed) if not c],
    }
