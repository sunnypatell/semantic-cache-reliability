# SPDX-License-Identifier: LicenseRef-Proprietary
# Copyright (c) 2026 Sunny Patel. All rights reserved.
"""Correctness tests for the statistics, metrics, and calibration cores.

These pin the behavior the paper depends on. They use synthetic data with known
structure so a regression in, say, the bootstrap or the false-hit-rate definition
fails loudly before it ever touches a real model.
"""
from __future__ import annotations

import numpy as np

from cacherel import calibration as C
from cacherel import metrics as M
from cacherel import stats as S


def _separable(n=2000, sep=1.5, base_rate=0.4, seed=0):
    """Synthetic (y, s): positives drawn higher than negatives, partially overlapping."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < base_rate).astype(int)
    s = np.where(y == 1, rng.normal(sep, 1.0, n), rng.normal(0.0, 1.0, n))
    # squash to a cosine-like [-1, 1] range
    s = np.tanh(s / 2.0)
    return y, s


def test_bca_ci_brackets_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(5.0, 2.0, 800)
    ci = S.bca_ci(x, np.mean, n_boot=1500, seed=2)
    assert ci.low < ci.estimate < ci.high
    assert ci.low < 5.0 < ci.high  # true mean inside a 95% CI (very likely)
    assert abs(ci.estimate - x.mean()) < 1e-9


def test_holm_and_bh_dominate_raw_and_are_monotone():
    p = [0.001, 0.01, 0.02, 0.2, 0.5]
    h = S.holm(p)
    bh = S.benjamini_hochberg(p)
    assert np.all(h >= np.array(p) - 1e-12)
    assert np.all(bh >= np.array(p) - 1e-12)
    assert np.all(h >= bh - 1e-12)  # Holm (FWER) at least as strict as BH (FDR)
    order = np.argsort(p)
    assert np.all(np.diff(h[order]) >= -1e-12)


def test_cohens_d_recovers_shift():
    rng = np.random.default_rng(3)
    x = rng.normal(1.0, 1.0, 5000)
    y = rng.normal(0.0, 1.0, 5000)
    d = S.cohens_d(x, y)
    assert abs(d - 1.0) < 0.1


def test_mcnemar_symmetric_is_one_and_skewed_is_small():
    assert abs(S.mcnemar_test(10, 10) - 1.0) < 1e-9
    assert S.mcnemar_test(30, 2) < 0.05


def test_metrics_on_separable_data():
    y, s = _separable(sep=2.0)
    assert M.pr_auc(y, s) > 0.85
    assert M.roc_auc(y, s) > 0.9
    # FHR is non-increasing as the threshold rises.
    taus = np.linspace(-0.9, 0.95, 40)
    fhr = np.array([M.fhr_at_threshold(y, s, t) for t in taus])
    fhr = fhr[~np.isnan(fhr)]
    assert np.all(np.diff(fhr) <= 1e-6)
    # Coverage is non-increasing as the threshold rises.
    cov = np.array([M.hit_rate_at_threshold(y, s, t) for t in taus])
    assert np.all(np.diff(cov) <= 1e-9)


def test_precision_at_recall_and_coverage_duality():
    y, s = _separable(sep=2.0)
    p70 = M.precision_at_recall(y, s, 0.7)
    p90 = M.precision_at_recall(y, s, 0.9)
    assert 0.0 <= p90 <= p70 <= 1.0  # demanding more recall cannot raise precision
    cov1 = M.coverage_at_max_fhr(y, s, 0.01)
    cov5 = M.coverage_at_max_fhr(y, s, 0.05)
    assert 0.0 <= cov1 <= cov5 <= 1.0  # tolerating more error cannot lower coverage


def test_cost_aware_threshold_beats_bad_fixed():
    y, s = _separable(sep=2.0)
    n = len(y)
    half = n // 2
    yv, sv = y[:half], s[:half]
    yt, st = y[half:], s[half:]
    c_fh = 10.0
    pol = C.cost_aware_threshold(yv, sv, c_fh)
    bad = C.fixed_threshold(-0.95)  # serve almost everything: many false hits
    e_cost = C.evaluate_policy(pol, yt, st, c_fh).expected_cost
    e_bad = C.evaluate_policy(bad, yt, st, c_fh).expected_cost
    assert e_cost < e_bad


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
