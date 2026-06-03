# SPDX-License-Identifier: LicenseRef-Proprietary
# Copyright (c) 2026 Sunny Patel. All rights reserved.
"""Cache-reliability metrics.

Conventions
-----------
Each candidate pair carries a similarity score ``s`` (higher means the cache judges
the prompts closer) and a binary label ``y``: ``y = 1`` means the pair is
*cache-equivalent* (serving the stored answer is safe) and ``y = 0`` means it is not.
A cache *hit* at threshold tau occurs when ``s >= tau``: the stored answer is served.

  * true hit  : hit and y = 1
  * false hit : hit and y = 0   (a silently wrong answer)

The headline reliability quantity is the false-hit rate among served queries,

    FHR(tau) = P(y = 0 | s >= tau) = FP / (TP + FP) = 1 - precision(tau),

reported alongside coverage (hit rate) so the cost of safety is explicit. Because
false hits are the rare, costly event, threshold-free separability is summarized by
PR-AUC first and ROC-AUC second.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _validate(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    if y.shape != s.shape:
        raise ValueError("labels and scores must have the same shape")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("labels must be binary (0/1)")
    return y, s


def fhr_at_threshold(y: np.ndarray, s: np.ndarray, tau: float) -> float:
    """False-hit rate among queries served at threshold ``tau``.

    Returns ``nan`` when nothing is served (no hits), since FHR is undefined there.
    """
    y, s = _validate(y, s)
    hit = s >= tau
    n_hit = int(hit.sum())
    if n_hit == 0:
        return float("nan")
    false_hits = int(np.sum(hit & (y == 0)))
    return false_hits / n_hit


def hit_rate_at_threshold(y: np.ndarray, s: np.ndarray, tau: float) -> float:
    """Fraction of queries served from cache at threshold ``tau`` (coverage)."""
    y, s = _validate(y, s)
    return float(np.mean(s >= tau))


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    fhr: float
    hit_rate: float
    precision: float
    recall: float
    f1: float


def operating_point(y: np.ndarray, s: np.ndarray, tau: float) -> OperatingPoint:
    """Full operating-point summary at a fixed threshold."""
    y, s = _validate(y, s)
    hit = s >= tau
    tp = int(np.sum(hit & (y == 1)))
    fp = int(np.sum(hit & (y == 0)))
    fn = int(np.sum(~hit & (y == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    fhr = (1.0 - precision) if (tp + fp) > 0 else float("nan")
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return OperatingPoint(
        threshold=float(tau),
        fhr=fhr,
        hit_rate=float(np.mean(hit)),
        precision=precision,
        recall=recall,
        f1=f1,
    )


def pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision).

    Positives are cache-equivalent pairs. PR-AUC is the headline threshold-free
    metric because it is informative under the class imbalance typical of cache
    traffic, where the decisive minority is the high-similarity non-equivalent pair.
    """
    y, s = _validate(y, s)
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(average_precision_score(y, s))


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Area under the ROC curve (threshold-free separability, secondary metric)."""
    y, s = _validate(y, s)
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(roc_auc_score(y, s))


def precision_at_recall(y: np.ndarray, s: np.ndarray, target_recall: float) -> float:
    """Highest precision achievable while recall is at least ``target_recall``.

    This is the operating-point view a practitioner cares about: "if I insist on
    reusing at least this fraction of genuinely reusable answers, how clean can the
    cache be?" Returns ``nan`` if the recall target is unreachable.
    """
    y, s = _validate(y, s)
    pos = int(y.sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")  # stable, high score first
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    recall = tp / pos
    precision = np.where((tp + fp) > 0, tp / np.maximum(tp + fp, 1), 0.0)
    mask = recall >= target_recall
    if not np.any(mask):
        return float("nan")
    return float(np.max(precision[mask]))


def coverage_at_max_fhr(y: np.ndarray, s: np.ndarray, max_fhr: float) -> float:
    """Maximum coverage (hit rate) achievable while FHR stays at or below ``max_fhr``.

    The dual of :func:`precision_at_recall`: "if the cache must serve wrong answers
    no more than this often, how many queries can it serve?" Returns 0.0 if no
    threshold meets the constraint with any hits.
    """
    y, s = _validate(y, s)
    n = y.size
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    served = tp + fp
    fhr = np.where(served > 0, fp / np.maximum(served, 1), 0.0)
    ok = fhr <= max_fhr
    if not np.any(ok):
        return 0.0
    return float(np.max(served[ok]) / n)
