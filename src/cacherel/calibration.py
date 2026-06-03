# SPDX-License-Identifier: LicenseRef-Proprietary
# Copyright (c) 2026 Sunny Patel. All rights reserved.
"""Threshold policies for semantic caching, including a cost-aware calibrator.

A cache must choose a single number: the similarity threshold tau above which it
serves a stored answer. Existing practice picks tau to hit a fixed similarity value,
or, as in verified caching, learns a per-prompt threshold online from correctness
feedback. Both optimize a symmetric notion of correctness. They ignore that the cost
of a false hit is not the cost of a miss: serving a wrong answer can be far more
damaging than recomputing one, and how much more is task-dependent.

We model that asymmetry explicitly. With every served-and-correct hit saving one model
call, every miss costing one recomputation, and every false hit costing ``c_fh`` calls
(a penalty that the downstream experiment estimates per task), the expected cost per
query at threshold tau is

    C(tau) = [ c_fh * FP(tau) + (TN(tau) + FN(tau)) ] / N,

measured against the always-recompute baseline cost of 1 per query. The cost-aware
threshold minimizes C. Because the penalty ``c_fh`` is the only domain-specific input
and it is measured rather than guessed, the resulting policy transfers across encoders
and domains without per-item online feedback.

This module provides the cost model, the policies, and a uniform evaluation so the
paper can compare them on identical test data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def expected_cost_per_query(y: np.ndarray, s: np.ndarray, tau: float, c_fh: float) -> float:
    """Expected cost per query at threshold ``tau`` under the false-hit penalty ``c_fh``.

    Cost is in units of model calls relative to always recomputing (which costs 1/query):
    a correct hit saves the call (cost 0), a miss recomputes (cost 1), a false hit costs
    ``c_fh``. Lower is better; values below 1 mean the cache is a net win.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    hit = s >= tau
    tp = float(np.sum(hit & (y == 1)))
    fp = float(np.sum(hit & (y == 0)))
    miss = float(np.sum(~hit))  # TN + FN: recomputed
    n = float(y.size)
    return (c_fh * fp + miss) / n


def cost_curve(y, s, taus: np.ndarray, c_fh: float) -> np.ndarray:
    return np.array([expected_cost_per_query(y, s, t, c_fh) for t in taus], dtype=float)


@dataclass(frozen=True)
class Policy:
    name: str
    threshold: float


def fixed_threshold(tau0: float) -> Policy:
    """A single hand-set threshold applied everywhere (the common default)."""
    return Policy(name=f"fixed@{tau0:g}", threshold=float(tau0))


def fhr_target_threshold(y_val, s_val, target_fhr: float, *, grid: np.ndarray | None = None) -> Policy:
    """Lowest threshold whose validation false-hit rate stays at or below ``target_fhr``.

    This stands in for correctness-bounding policies such as verified caching: it
    maximizes coverage subject to a hard reliability ceiling, but it is blind to the
    relative cost of a miss.
    """
    from .metrics import fhr_at_threshold, hit_rate_at_threshold

    if grid is None:
        grid = np.linspace(0.0, 1.0, 201)
    best = None
    for t in grid:
        f = fhr_at_threshold(y_val, s_val, t)
        if np.isnan(f) or f <= target_fhr:
            cov = hit_rate_at_threshold(y_val, s_val, t)
            if best is None or cov > best[1]:
                best = (t, cov)
    tau = best[0] if best is not None else 1.0
    return Policy(name=f"fhr<= {target_fhr:g}", threshold=float(tau))


def cost_aware_threshold(y_val, s_val, c_fh: float, *, grid: np.ndarray | None = None) -> Policy:
    """Threshold minimizing validation expected cost under false-hit penalty ``c_fh``."""
    if grid is None:
        grid = np.linspace(0.0, 1.0, 201)
    costs = cost_curve(y_val, s_val, grid, c_fh)
    tau = float(grid[int(np.argmin(costs))])
    return Policy(name=f"cost-aware(c_fh={c_fh:g})", threshold=tau)


@dataclass(frozen=True)
class PolicyEvaluation:
    policy: str
    threshold: float
    fhr: float
    coverage: float
    expected_cost: float


def evaluate_policy(policy: Policy, y_test, s_test, c_fh: float) -> PolicyEvaluation:
    """Apply a policy's threshold to held-out test data and report the trade-off."""
    from .metrics import fhr_at_threshold, hit_rate_at_threshold

    return PolicyEvaluation(
        policy=policy.name,
        threshold=policy.threshold,
        fhr=fhr_at_threshold(y_test, s_test, policy.threshold),
        coverage=hit_rate_at_threshold(y_test, s_test, policy.threshold),
        expected_cost=expected_cost_per_query(y_test, s_test, policy.threshold, c_fh),
    )
