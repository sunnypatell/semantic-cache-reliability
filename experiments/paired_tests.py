# SPDX-License-Identifier: LicenseRef-Proprietary
# Copyright (c) 2026 Sunny Patel. All rights reserved.
"""Paired significance tests for the cross-encoder separability claims.

The headline claim ("on the adversarial domain a lexical baseline outscores every
neural encoder") must rest on a paired test, not on overlapping marginal intervals.
Every encoder scores the *same* labeled pairs, so we resample pair indices jointly and
recompute both encoders' AUCs on each resample. This captures the positive correlation
between encoders that a marginal interval ignores, and yields a bootstrap p-value for the
difference. p-values are corrected across the family of comparisons with Benjamini-Hochberg.

Run:  python experiments/paired_tests.py
Writes: results/paired_tests.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cacherel.metrics import pr_auc, roc_auc  # noqa: E402
from src.cacherel.stats import benjamini_hochberg  # noqa: E402

RELIABILITY = ROOT / "results" / "reliability"
NEURAL = [
    "all-MiniLM-L6-v2", "all-mpnet-base-v2", "e5-small-v2",
    "e5-large-v2", "bge-base-en-v1.5", "bge-large-en-v1.5",
]
LEXICAL = "tfidf-lexical"
N_BOOT = 10000
SEED = 0


def load_labels_scores(dataset: str, encoder: str) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct aligned (label, score) vectors from the saved pos/neg score arrays.

    Every encoder iterates the dataset in the same order, so concatenating the positive
    then the negative scores yields vectors whose index i refers to the same pair across
    encoders, which is what makes the bootstrap below paired.
    """
    d = json.loads((RELIABILITY / f"{dataset}__{encoder}.json").read_text(encoding="utf-8"))
    pos = np.asarray(d["curves"]["scores"]["pos"], dtype=float)
    neg = np.asarray(d["curves"]["scores"]["neg"], dtype=float)
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return labels, scores


def paired_auc_test(labels: np.ndarray, s_a: np.ndarray, s_b: np.ndarray,
                    metric, n_boot: int = N_BOOT, seed: int = SEED):
    """Paired bootstrap on a set-level metric (PR-AUC or ROC-AUC).

    Returns observed diff (a - b), its 95% percentile CI, and a two-sided bootstrap
    p-value for diff = 0.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    obs = metric(labels, s_a) - metric(labels, s_b)
    diffs = np.empty(n_boot)
    k = 0
    while k < n_boot:
        idx = rng.integers(0, n, n)
        yl = labels[idx]
        if yl.sum() == 0 or yl.sum() == n:  # degenerate resample, redraw
            continue
        diffs[k] = metric(yl, s_a[idx]) - metric(yl, s_b[idx])
        k += 1
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided bootstrap p: reflect the smaller tail mass about zero
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    p = min(p, 1.0)
    return obs, float(lo), float(hi), float(p)


def main() -> None:
    dataset = "paws"
    lab_lex, s_lex = load_labels_scores(dataset, LEXICAL)
    rows = []
    for enc in NEURAL:
        lab_n, s_n = load_labels_scores(dataset, enc)
        assert np.array_equal(lab_lex, lab_n), f"label order mismatch for {enc}"
        for mname, metric in (("pr_auc", pr_auc), ("roc_auc", roc_auc)):
            obs, lo, hi, p = paired_auc_test(lab_lex, s_lex, s_n, metric)
            rows.append({
                "dataset": dataset, "metric": mname, "lexical": LEXICAL, "neural": enc,
                "lexical_auc": metric(lab_lex, s_lex), "neural_auc": metric(lab_n, s_n),
                "diff": obs, "ci_lo": lo, "ci_hi": hi, "p_raw": p,
            })

    # BH-correct within each metric family (6 comparisons each).
    for mname in ("pr_auc", "roc_auc"):
        fam = [r for r in rows if r["metric"] == mname]
        p_adj = benjamini_hochberg([r["p_raw"] for r in fam])
        for r, pa in zip(fam, p_adj):
            r["p_bh"] = float(pa)

    import csv
    out = ROOT / "results" / "paired_tests.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"[paired] PAWS lexical (TF-IDF) vs each neural encoder, {N_BOOT} bootstrap resamples\n")
    for mname in ("pr_auc", "roc_auc"):
        print(f"== {mname} ==")
        for r in [r for r in rows if r["metric"] == mname]:
            star = "*" if r["p_bh"] < 0.05 else " "
            print(f"  vs {r['neural']:<20} lex={r['lexical_auc']:.3f} neu={r['neural_auc']:.3f} "
                  f"diff=+{r['diff']:.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] "
                  f"p_bh={r['p_bh']:.4f} {star}")
        worst = max([r for r in rows if r["metric"] == mname], key=lambda r: r["neural_auc"])
        print(f"  -> strongest neural on {mname}: {worst['neural']} ({worst['neural_auc']:.3f}); "
              f"lexical {worst['lexical_auc']:.3f}, diff +{worst['diff']:.3f}, BH p={worst['p_bh']:.4f}\n")
    print(f"[paired] -> {out}")


if __name__ == "__main__":
    main()
