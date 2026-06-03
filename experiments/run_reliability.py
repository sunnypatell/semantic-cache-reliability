# SPDX-License-Identifier: LicenseRef-Proprietary
# Copyright (c) 2026 Sunny Patel. All rights reserved.
"""Core reliability experiment: false-hit economics across models, thresholds, domains.

For every labeled pair set and every encoder, this computes the threshold-free
separability (PR-AUC, ROC-AUC) with bootstrap confidence intervals, the false-hit /
coverage trade-off curve, and the operating points a practitioner reasons about
(precision at fixed recall; coverage under a false-hit ceiling). Results are written
to ``results/reliability/`` as one JSON per (dataset, encoder), plus a flat summary CSV.

Run (from the repository root, with the project installed: ``pip install -e .``):

    python experiments/run_reliability.py --max-pairs 2500 --n-boot 2000
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cacherel.data import available_datasets, load_pairs
from cacherel.embeddings import DEFAULT_MODEL_SPECS, LexicalEncoder, SentenceEncoder
from cacherel.evaluate import reliability_report

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "reliability"
PROCESSED = ROOT / "data" / "processed"


def _save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _summary_row(rep) -> dict:
    pr, roc = rep.pr_auc, rep.roc_auc
    row = {
        "dataset": rep.dataset,
        "encoder": rep.encoder,
        "n_pairs": rep.n_pairs,
        "base_rate": round(rep.base_rate, 4),
        "pr_auc": round(pr["estimate"], 4),
        "pr_auc_lo": round(pr["low"], 4),
        "pr_auc_hi": round(pr["high"], 4),
        "roc_auc": round(roc["estimate"], 4),
        "roc_auc_lo": round(roc["low"], 4),
        "roc_auc_hi": round(roc["high"], 4),
    }
    for r, ci in rep.precision_at_recall.items():
        row[f"prec@rec{r}"] = round(ci["estimate"], 4)
    for mf, ci in rep.coverage_at_max_fhr.items():
        row[f"cov@fhr{mf}"] = round(ci["estimate"], 4)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=available_datasets())
    ap.add_argument("--max-pairs", type=int, default=2500)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-lexical", action="store_true")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Subset of Hugging Face model names; defaults to the pinned roster.")
    args = ap.parse_args()

    specs = DEFAULT_MODEL_SPECS
    if args.models is not None:
        specs = [s for s in DEFAULT_MODEL_SPECS if s["model_name"] in args.models]

    # Load each labeled pair set once and record exactly what was used.
    pairsets = {}
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        ps = load_pairs(name, max_pairs=args.max_pairs, seed=args.seed)
        pairsets[name] = ps
        pd.DataFrame({"a": ps.a, "b": ps.b, "y": ps.y}).to_csv(
            PROCESSED / f"{name}_pairs.csv", index=False, encoding="utf-8"
        )
        print(f"[data] {name}: n={len(ps)} base_rate={ps.base_rate:.3f}")

    summary: list[dict] = []
    summary_path = ROOT / "results" / "reliability_summary.csv"

    def flush_summary() -> None:
        # Persist after every report so an interruption never loses completed work.
        pd.DataFrame(summary).to_csv(summary_path, index=False, encoding="utf-8")

    def process(enc, label: str) -> None:
        for name, ps in pairsets.items():
            t0 = time.time()
            try:
                rep = reliability_report(enc, ps, n_boot=args.n_boot, seed=args.seed)
            except Exception as exc:  # keep the run alive; record and move on
                print(f"[skip] {label:28s} x {name:5s} ERROR: {exc}")
                continue
            _save_json(rep.to_json(), RESULTS / f"{name}__{rep.encoder}.json")
            summary.append(_summary_row(rep))
            flush_summary()
            print(
                f"[run] {label:28s} x {name:5s} "
                f"PR-AUC={rep.pr_auc['estimate']:.3f} "
                f"ROC-AUC={rep.roc_auc['estimate']:.3f} "
                f"({time.time() - t0:.1f}s)"
            )

    for spec in specs:
        try:
            enc = SentenceEncoder(**spec)
        except Exception as exc:
            print(f"[skip] could not load {spec['model_name']}: {exc}")
            continue
        process(enc, enc.name)
        del enc
        gc.collect()

    if not args.no_lexical:
        for name, ps in pairsets.items():
            enc = LexicalEncoder()
            try:
                rep = reliability_report(enc, ps, n_boot=args.n_boot, seed=args.seed)
            except Exception as exc:
                print(f"[skip] tfidf-lexical x {name}: {exc}")
                continue
            _save_json(rep.to_json(), RESULTS / f"{name}__{rep.encoder}.json")
            summary.append(_summary_row(rep))
            flush_summary()
            print(f"[run] {'tfidf-lexical':28s} x {name:5s} "
                  f"PR-AUC={rep.pr_auc['estimate']:.3f} ({time.time() - t0:.1f}s)")

    flush_summary()
    print(f"\n[done] {len(summary)} reports -> {summary_path}")
    if summary:
        print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
