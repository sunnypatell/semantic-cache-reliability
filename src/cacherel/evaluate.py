# SPDX-License-Identifier: LicenseRef-Proprietary
# Copyright (c) 2026 Sunny Patel. All rights reserved.
"""Orchestration: from labeled pairs and an encoder to a reliability report.

This module turns a (label, similarity) sample into the numbers and curves the paper
reports, each headline quantity carrying a bias-corrected and accelerated bootstrap
confidence interval. Curves (PR, ROC, the false-hit/coverage trade-off) are returned
as plain arrays so the figure layer can render them without recomputation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve as _sk_roc_curve

from . import metrics as M
from . import stats as S
from .data import PairSet
from .embeddings import Encoder, LexicalEncoder


def score_pairs(encoder: Encoder, pairs: PairSet) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y, s)``: equivalence labels and the encoder's pair similarities."""
    if isinstance(encoder, LexicalEncoder):
        encoder.fit(pairs.corpus())
    s = np.asarray(encoder.pair_similarity(pairs.a, pairs.b), dtype=float)
    y = np.asarray(pairs.y, dtype=int)
    return y, s


def _ci_for(y, s, fn, *, n_boot: int, seed: int, max_jackknife: int | None = None) -> S.CI:
    """Bootstrap CI for a metric ``fn(y, s)`` by resampling pairs."""
    n = len(y)
    data = np.arange(n)

    def stat(idx: np.ndarray) -> float:
        return float(fn(y[idx], s[idx]))

    return S.bca_ci(data, stat, n_boot=n_boot, seed=seed, max_jackknife=max_jackknife)


def fhr_coverage_curve(
    y: np.ndarray, s: np.ndarray, taus: np.ndarray
) -> dict[str, np.ndarray]:
    """False-hit rate and coverage (hit rate) across a grid of thresholds."""
    fhr = np.array([M.fhr_at_threshold(y, s, t) for t in taus], dtype=float)
    cov = np.array([M.hit_rate_at_threshold(y, s, t) for t in taus], dtype=float)
    return {"tau": np.asarray(taus, dtype=float), "fhr": fhr, "coverage": cov}


def pr_curve(y: np.ndarray, s: np.ndarray) -> dict[str, np.ndarray]:
    precision, recall, thr = precision_recall_curve(y, s)
    return {"precision": precision, "recall": recall, "thresholds": thr}


def roc_curve(y: np.ndarray, s: np.ndarray) -> dict[str, np.ndarray]:
    fpr, tpr, thr = _sk_roc_curve(y, s)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thr}


@dataclass
class ReliabilityReport:
    dataset: str
    encoder: str
    n_pairs: int
    base_rate: float
    pr_auc: dict
    roc_auc: dict
    precision_at_recall: dict  # recall target -> CI dict
    coverage_at_max_fhr: dict  # max-fhr target -> CI dict
    curves: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        out = asdict(self)
        # Curves are arrays; convert for JSON serialization at save time.
        out["curves"] = {
            k: {kk: np.asarray(vv).tolist() for kk, vv in v.items()}
            for k, v in self.curves.items()
        }
        return out


def reliability_report(
    encoder: Encoder,
    pairs: PairSet,
    *,
    taus: np.ndarray | None = None,
    recall_targets: tuple[float, ...] = (0.7, 0.8, 0.9),
    max_fhr_targets: tuple[float, ...] = (0.01, 0.05),
    n_boot: int = 2000,
    seed: int = 0,
    max_jackknife: int | None = 1500,
) -> ReliabilityReport:
    """Full reliability report for one encoder on one labeled pair set."""
    y, s = score_pairs(encoder, pairs)
    if taus is None:
        taus = np.linspace(0.0, 1.0, 51)

    pr = _ci_for(y, s, M.pr_auc, n_boot=n_boot, seed=seed, max_jackknife=max_jackknife)
    roc = _ci_for(y, s, M.roc_auc, n_boot=n_boot, seed=seed, max_jackknife=max_jackknife)

    par = {}
    for r in recall_targets:
        par[f"{r:.2f}"] = _ci_for(
            y, s, lambda yy, ss, r=r: M.precision_at_recall(yy, ss, r),
            n_boot=n_boot, seed=seed, max_jackknife=max_jackknife,
        ).__dict__

    cov = {}
    for mf in max_fhr_targets:
        cov[f"{mf:.2f}"] = _ci_for(
            y, s, lambda yy, ss, mf=mf: M.coverage_at_max_fhr(yy, ss, mf),
            n_boot=n_boot, seed=seed, max_jackknife=max_jackknife,
        ).__dict__

    curves = {
        "fhr_coverage": fhr_coverage_curve(y, s, taus),
        "pr": pr_curve(y, s),
        "roc": roc_curve(y, s),
        "scores_pos": s[y == 1],
        "scores_neg": s[y == 0],
    }
    # Store score arrays under a dict wrapper so to_json handles them uniformly.
    curves["scores"] = {"pos": curves.pop("scores_pos"), "neg": curves.pop("scores_neg")}

    return ReliabilityReport(
        dataset=pairs.name,
        encoder=getattr(encoder, "name", encoder.__class__.__name__),
        n_pairs=len(y),
        base_rate=float(np.mean(y)),
        pr_auc=pr.__dict__,
        roc_auc=roc.__dict__,
        precision_at_recall=par,
        coverage_at_max_fhr=cov,
        curves=curves,
    )
