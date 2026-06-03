# SPDX-License-Identifier: LicenseRef-Proprietary
# Copyright (c) 2026 Sunny Patel. All rights reserved.
"""Calibration experiment: cost-aware thresholding versus baselines, and transfer.

Reads the per-(dataset, encoder) score arrays saved by ``run_reliability.py`` (no
re-encoding needed, since every threshold quantity depends only on the per-class score
multisets), splits each into a calibration and a test half, and compares threshold
policies on held-out data:

  fixed@0.80   : a hand-set similarity threshold, the common default.
  fhr<=0.05    : the lowest threshold whose calibration false-hit rate stays under 5%
                 (a correctness-bounding policy, in the spirit of verified caching).
  cost-aware   : our policy, which minimizes expected cost under a false-hit penalty.

It also measures transfer: calibrate a policy on one domain and apply it to another,
reporting the cost gap against calibrating in-domain.

Run after run_reliability.py:  python experiments/run_calibration.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cacherel import calibration as C

ROOT = Path(__file__).resolve().parents[1]
RELIABILITY = ROOT / "results" / "reliability"


def load_scores(path: Path) -> tuple[str, str, np.ndarray, np.ndarray]:
    """Reconstruct (y, s) from a reliability report's saved per-class score arrays."""
    obj = json.loads(path.read_text(encoding="utf-8"))
    pos = np.asarray(obj["curves"]["scores"]["pos"], dtype=float)
    neg = np.asarray(obj["curves"]["scores"]["neg"], dtype=float)
    y = np.concatenate([np.ones(pos.size, int), np.zeros(neg.size, int)])
    s = np.concatenate([pos, neg])
    return obj["dataset"], obj["encoder"], y, s


def split(y: np.ndarray, s: np.ndarray, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(y.size)
    half = y.size // 2
    cal, test = idx[:half], idx[half:]
    return (y[cal], s[cal]), (y[test], s[test])


def main() -> None:
    reports = sorted(RELIABILITY.glob("*.json"))
    if not reports:
        raise SystemExit("No reliability reports found; run run_reliability.py first.")

    c_fh_values = [5.0, 20.0]
    rows: list[dict] = []
    scores: dict[tuple[str, str], tuple] = {}

    for path in reports:
        dataset, encoder, y, s = load_scores(path)
        scores[(dataset, encoder)] = (y, s)
        (yc, sc), (yt, st) = split(y, s)
        for c_fh in c_fh_values:
            policies = [
                C.fixed_threshold(0.80),
                C.fhr_target_threshold(yc, sc, 0.05),
                C.cost_aware_threshold(yc, sc, c_fh),
            ]
            for pol in policies:
                ev = C.evaluate_policy(pol, yt, st, c_fh)
                rows.append({
                    "dataset": dataset, "encoder": encoder, "c_fh": c_fh,
                    "policy": ev.policy, "threshold": round(ev.threshold, 3),
                    "fhr": round(ev.fhr, 4) if not np.isnan(ev.fhr) else None,
                    "coverage": round(ev.coverage, 4),
                    "expected_cost": round(ev.expected_cost, 4),
                })

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "calibration_summary.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"[done] calibration -> {out} ({len(df)} rows)")

    # Transfer: for each encoder, calibrate cost-aware on domain A, apply to domain B.
    transfer_rows: list[dict] = []
    encoders = sorted({e for (_, e) in scores})
    domains = sorted({d for (d, _) in scores})
    c_fh = 20.0
    for enc in encoders:
        for da in domains:
            if (da, enc) not in scores:
                continue
            (yc_a, sc_a), _ = split(*scores[(da, enc)])
            pol = C.cost_aware_threshold(yc_a, sc_a, c_fh)
            for db in domains:
                if (db, enc) not in scores:
                    continue
                _, (yt_b, st_b) = split(*scores[(db, enc)])
                ev = C.evaluate_policy(pol, yt_b, st_b, c_fh)
                # In-domain reference for db.
                (yc_b, sc_b), (yt_b2, st_b2) = split(*scores[(db, enc)])
                pol_b = C.cost_aware_threshold(yc_b, sc_b, c_fh)
                ev_b = C.evaluate_policy(pol_b, yt_b2, st_b2, c_fh)
                transfer_rows.append({
                    "encoder": enc, "calibrated_on": da, "applied_to": db,
                    "threshold": round(pol.threshold, 3),
                    "transfer_cost": round(ev.expected_cost, 4),
                    "in_domain_cost": round(ev_b.expected_cost, 4),
                    "cost_gap": round(ev.expected_cost - ev_b.expected_cost, 4),
                })
    tdf = pd.DataFrame(transfer_rows)
    tout = ROOT / "results" / "calibration_transfer.csv"
    tdf.to_csv(tout, index=False, encoding="utf-8")
    print(f"[done] transfer -> {tout} ({len(tdf)} rows)")


if __name__ == "__main__":
    main()
