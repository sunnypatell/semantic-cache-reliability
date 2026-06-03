# SPDX-License-Identifier: LicenseRef-Proprietary
# Copyright (c) 2026 Sunny Patel. All rights reserved.
"""One-off: compute the lexical-control reports and rebuild the summary from all reports.

Used to recover from a crash in the lexical loop of an earlier run_reliability.py without
re-encoding the six neural models. Safe to run repeatedly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cacherel.data import load_pairs
from cacherel.embeddings import LexicalEncoder
from cacherel.evaluate import reliability_report

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "reliability"

for name in ["qqp", "mrpc", "paws"]:
    ps = load_pairs(name, max_pairs=2500, seed=0)
    rep = reliability_report(LexicalEncoder(), ps, n_boot=1500, seed=0)
    (RESULTS / f"{name}__{rep.encoder}.json").write_text(json.dumps(rep.to_json(), indent=2), encoding="utf-8")
    print(f"[lexical] {name}: PR-AUC={rep.pr_auc['estimate']:.3f}")

rows = []
for p in sorted(RESULTS.glob("*.json")):
    o = json.loads(p.read_text(encoding="utf-8"))
    pr, roc = o["pr_auc"], o["roc_auc"]
    row = {
        "dataset": o["dataset"], "encoder": o["encoder"], "n_pairs": o["n_pairs"],
        "base_rate": round(o["base_rate"], 4),
        "pr_auc": round(pr["estimate"], 4), "pr_auc_lo": round(pr["low"], 4), "pr_auc_hi": round(pr["high"], 4),
        "roc_auc": round(roc["estimate"], 4), "roc_auc_lo": round(roc["low"], 4), "roc_auc_hi": round(roc["high"], 4),
    }
    for r, ci in o["precision_at_recall"].items():
        row[f"prec@rec{r}"] = round(ci["estimate"], 4)
    for mf, ci in o["coverage_at_max_fhr"].items():
        row[f"cov@fhr{mf}"] = round(ci["estimate"], 4)
    rows.append(row)

pd.DataFrame(rows).to_csv(ROOT / "results" / "reliability_summary.csv", index=False, encoding="utf-8")
print(f"[summary] rebuilt with {len(rows)} rows -> results/reliability_summary.csv")
