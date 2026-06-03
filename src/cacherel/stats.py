# SPDX-License-Identifier: LicenseRef-Proprietary
# Copyright (c) 2026 Sunny Patel. All rights reserved.
"""Uncertainty quantification and inference for cache-reliability experiments.

Design choices:
  * Confidence intervals use the bias-corrected and accelerated (BCa) bootstrap,
    which corrects for skew and bias that the percentile bootstrap ignores. This
    matters because the false-hit rate is a rare-event proportion whose sampling
    distribution is skewed.
  * Comparisons between two methods evaluated on the *same* items use a paired
    bootstrap over items, preserving the pairing and raising power.
  * Families of comparisons are corrected with Holm (controls FWER) or
    Benjamini-Hochberg (controls FDR), never left uncorrected.
  * Effect sizes (Cohen's h for proportions, Cohen's d for means) accompany every
    p-value, because significance without magnitude is uninformative.

All functions are deterministic given a seed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import stats as _sps


@dataclass(frozen=True)
class CI:
    """A point estimate with a confidence interval."""

    estimate: float
    low: float
    high: float
    level: float = 0.95

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.estimate:.4f} [{self.low:.4f}, {self.high:.4f}] ({self.level:.0%} CI)"


def _percentile_of(values: np.ndarray, x: float) -> float:
    """Fraction of ``values`` strictly less than ``x`` (for the BCa bias term)."""
    values = np.asarray(values, dtype=float)
    return float(np.mean(values < x))


def bca_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
    max_jackknife: int | None = None,
) -> CI:
    """Bias-corrected and accelerated bootstrap CI for a one-sample statistic.

    Parameters
    ----------
    data : array of shape (n,) or (n, k); resampling is over the first axis.
    statistic : maps a resample of ``data`` to a scalar.
    n_boot : number of bootstrap resamples.
    alpha : two-sided error (0.05 -> 95% CI).
    seed : RNG seed for reproducibility.
    max_jackknife : if set and ``n`` exceeds it, the acceleration term is estimated
        from a random subsample of leave-one-out replicates rather than all ``n``.
        The acceleration is small and stable at large ``n``, so this trades a
        negligible bias for a large speedup on metrics like AUC.

    Implements Efron's BCa: the bias-correction ``z0`` from the fraction of bootstrap
    replicates below the observed estimate, and the acceleration ``a`` from the
    jackknife skewness of the statistic. Non-finite replicates are dropped so that a
    rare degenerate resample (for example, one missing a class) cannot poison the CI.
    """
    data = np.asarray(data)
    n = data.shape[0]
    if n < 2:
        raise ValueError("BCa bootstrap needs at least two observations.")
    rng = np.random.default_rng(seed)

    theta_hat = float(statistic(data))

    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = statistic(data[idx])
    boot = boot[np.isfinite(boot)]
    if boot.size < max(10, n_boot // 2):
        return CI(estimate=theta_hat, low=float("nan"), high=float("nan"), level=1.0 - alpha)

    # Bias-correction z0.
    prop = _percentile_of(boot, theta_hat)
    prop = min(max(prop, 1.0 / (n_boot + 1)), 1.0 - 1.0 / (n_boot + 1))
    z0 = _sps.norm.ppf(prop)

    # Acceleration a from the (optionally subsampled) jackknife.
    all_idx = np.arange(n)
    if max_jackknife is not None and n > max_jackknife:
        positions = rng.choice(n, size=max_jackknife, replace=False)
    else:
        positions = all_idx
    jack = np.array([statistic(data[all_idx != i]) for i in positions], dtype=float)
    jack = jack[np.isfinite(jack)]
    if jack.size >= 2:
        jack_mean = jack.mean()
        diff = jack_mean - jack
        denom = 6.0 * (np.sum(diff ** 2) ** 1.5)
        a = float(np.sum(diff ** 3) / denom) if denom != 0 else 0.0
    else:
        a = 0.0

    z_alpha_lo = _sps.norm.ppf(alpha / 2.0)
    z_alpha_hi = _sps.norm.ppf(1.0 - alpha / 2.0)

    def _adjust(z: float) -> float:
        return _sps.norm.cdf(z0 + (z0 + z) / (1.0 - a * (z0 + z)))

    p_lo = float(np.clip(_adjust(z_alpha_lo), 0.0, 1.0))
    p_hi = float(np.clip(_adjust(z_alpha_hi), 0.0, 1.0))
    low = float(np.quantile(boot, p_lo))
    high = float(np.quantile(boot, p_hi))
    return CI(estimate=theta_hat, low=low, high=high, level=1.0 - alpha)


def paired_bootstrap_diff(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[CI, float]:
    """Paired bootstrap for the mean difference ``a - b`` over shared items.

    Returns the CI for the mean difference and a two-sided bootstrap p-value for
    the null that the mean difference is zero. ``values_a`` and ``values_b`` are
    per-item scores (e.g., correct/incorrect indicators) on the same items.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Paired bootstrap requires equal-length, aligned inputs.")
    diff = a - b
    n = diff.shape[0]
    rng = np.random.default_rng(seed)

    obs = float(diff.mean())
    boot = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[k] = diff[idx].mean()

    low = float(np.quantile(boot, alpha / 2.0))
    high = float(np.quantile(boot, 1.0 - alpha / 2.0))
    # Two-sided p-value: reflect the bootstrap distribution about zero.
    centered = boot - obs
    p = float((np.sum(np.abs(centered) >= abs(obs)) + 1) / (n_boot + 1))
    return CI(estimate=obs, low=low, high=high, level=1.0 - alpha), p


def mcnemar_test(b: int, c: int, *, exact: bool | None = None) -> float:
    """McNemar's test for paired binary outcomes.

    ``b`` and ``c`` are the discordant counts (A right & B wrong; A wrong & B
    right). Uses the exact binomial test when discordant counts are small,
    otherwise the chi-square approximation with continuity correction.
    """
    n = b + c
    if n == 0:
        return 1.0
    if exact is None:
        exact = n < 25
    if exact:
        k = min(b, c)
        return float(min(1.0, 2.0 * _sps.binom.cdf(k, n, 0.5)))
    chi2 = (abs(b - c) - 1.0) ** 2 / n
    return float(_sps.chi2.sf(chi2, df=1))


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (controls family-wise error rate)."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (controls false discovery rate)."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        val = p[idx] * m / (rank + 1)
        running = min(running, val)
        adj[idx] = min(1.0, running)
    return adj


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for the difference between two proportions."""
    phi1 = 2.0 * np.arcsin(np.sqrt(np.clip(p1, 0.0, 1.0)))
    phi2 = 2.0 * np.arcsin(np.sqrt(np.clip(p2, 0.0, 1.0)))
    return float(phi1 - phi2)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d with pooled standard deviation for two independent samples."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        return float("nan")
    sp2 = ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    if sp2 == 0:
        return 0.0
    return float((x.mean() - y.mean()) / np.sqrt(sp2))
