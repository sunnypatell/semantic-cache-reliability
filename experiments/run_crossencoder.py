# SPDX-License-Identifier: LicenseRef-Proprietary
# Copyright (c) 2026 Sunny Patel. All rights reserved.
"""Cross-encoder verifier baseline.

A bi-encoder must commit each prompt to a single vector before it sees its neighbor, so it
cannot react to the specific other prompt; that is exactly the blindness the adversarial
domain exploits. A cross-encoder instead reads both prompts jointly and scores their
relation directly. It cannot index a cache (it is O(N) per query, with nothing to put in an
ANN structure), but it is the natural *verification* layer the paper recommends: retrieve a
candidate with a bi-encoder, then confirm equivalence with the cross-encoder before serving.

This measures that verifier on the same three labeled domains, with the same metrics and
bootstrap intervals as every other method, so it slots into the separability table and the
score-overlap analysis. We use an off-the-shelf semantic-similarity cross-encoder (no fitting
on our pairs) so the comparison is honest rather than tuned.

Run:  python experiments/run_crossencoder.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cacherel.data import load_pairs  # noqa: E402
from cacherel.evaluate import reliability_report  # noqa: E402

RESULTS = ROOT / "results" / "reliability"
MODEL = "cross-encoder/stsb-roberta-base"  # off-the-shelf STS cross-encoder, not fitted here
ENCODER_NAME = "cross-encoder-stsb"
DATASETS = ["qqp", "mrpc", "paws"]


@dataclass
class CrossEncoderVerifier:
    """Encoder-protocol adapter: scores each pair by joint cross-attention.

    Implements ``pair_similarity`` so it drops into ``reliability_report`` exactly like a
    bi-encoder, but the score is a direct equivalence judgment over the pair, not a cosine
    between two independently-formed vectors.
    """

    model_name: str = MODEL
    batch_size: int = 32
    name: str = ENCODER_NAME
    _model: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self.model_name)

    def pair_similarity(self, a, b) -> np.ndarray:
        if len(a) != len(b):
            raise ValueError("pair_similarity expects aligned, equal-length inputs")
        scores = self._model.predict(
            list(zip(list(a), list(b))), batch_size=self.batch_size, show_progress_bar=False
        )
        # STS cross-encoder scores are ~[0,1]; clip so the threshold sweep in
        # reliability_report (taus over [0,1]) is well-defined. AUCs are rank-based and
        # unaffected by the clip.
        return np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)


def _summary_row(rep) -> dict:
    pr, roc = rep.pr_auc, rep.roc_auc
    row = {
        "dataset": rep.dataset, "encoder": rep.encoder, "n_pairs": rep.n_pairs,
        "base_rate": round(rep.base_rate, 4),
        "pr_auc": round(pr["estimate"], 4), "pr_auc_lo": round(pr["low"], 4),
        "pr_auc_hi": round(pr["high"], 4),
        "roc_auc": round(roc["estimate"], 4), "roc_auc_lo": round(roc["low"], 4),
        "roc_auc_hi": round(roc["high"], 4),
    }
    for r, ci in rep.precision_at_recall.items():
        row[f"prec@rec{r}"] = round(ci["estimate"], 4)
    for mf, ci in rep.coverage_at_max_fhr.items():
        row[f"cov@fhr{mf}"] = round(ci["estimate"], 4)
    return row


def main() -> None:
    enc = CrossEncoderVerifier()
    rows = []
    for name in DATASETS:
        ps = load_pairs(name, max_pairs=2500, seed=0)
        t0 = time.time()
        rep = reliability_report(enc, ps, n_boot=2000, seed=0)
        (RESULTS / f"{name}__{ENCODER_NAME}.json").write_text(
            json.dumps(rep.to_json(), indent=2), encoding="utf-8")
        rows.append(_summary_row(rep))
        print(f"[xenc] {name:5s} PR-AUC={rep.pr_auc['estimate']:.3f} "
              f"ROC-AUC={rep.roc_auc['estimate']:.3f} ({time.time()-t0:.1f}s)")

    # Merge into the flat summary CSV (replace any prior cross-encoder rows; keep order).
    summ = ROOT / "results" / "reliability_summary.csv"
    df = pd.read_csv(summ)
    df = df[df["encoder"] != ENCODER_NAME]
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    order = {d: i for i, d in enumerate(DATASETS)}
    df["_o"] = df["dataset"].map(lambda d: order.get(d, 99))
    df = df.sort_values(["_o", "encoder"]).drop(columns="_o")
    df.to_csv(summ, index=False, encoding="utf-8")
    print(f"[xenc] merged {len(rows)} rows into {summ}")


if __name__ == "__main__":
    main()
