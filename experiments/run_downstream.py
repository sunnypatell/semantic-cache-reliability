# SPDX-License-Identifier: LicenseRef-Proprietary
# Copyright (c) 2026 Sunny Patel. All rights reserved.
"""Downstream cost of a false hit: controlled fault injection on extractive QA.

A false hit serves the stored response of a non-equivalent earlier query. How much that
costs depends on where the cache sits. We measure two positions on a SQuAD reading task
with a local extractive reader (no external API, fully reproducible):

  answer cache  : the cache returns a final answer directly. A false hit returns the gold
                  answer of a *different* question, which the user receives verbatim.
  context cache : the cache returns the retrieved passage that a reader then answers from.
                  A false hit hands the reader a wrong passage; the reader may recover
                  (find nothing useful and abstain) or compound the error.

For each question we compute the reader's token F1 under the correct condition and under
each fault condition. Because each question is faulted independently with probability p,
the end-to-end F1 at injected false-hit rate p is the exact mixture
(1-p) * F1_correct + p * F1_fault, which we report for p in {1, 2, 5, 10}%. Three fault
models (wrong passage, truncated passage, and a different question's answer) show the
conclusion does not hinge on how the false hit is manufactured.

Run:  python experiments/run_downstream.py --n 400
"""
from __future__ import annotations

import argparse
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def best_f1(pred: str, golds: list[str]) -> float:
    return max((token_f1(pred, g) for g in golds), default=0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Downstream fault-injection on extractive QA.")
    ap.add_argument("--n", type=int, default=400, help="number of questions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="distilbert-base-cased-distilled-squad")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    ds = load_dataset("rajpurkar/squad", split="validation")
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)
    rows = [ds[int(i)] for i in idx]

    tok = AutoTokenizer.from_pretrained(args.model)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(args.model)
    qa_model.eval()

    def answer(question: str, context: str) -> str:
        enc = tok(question, context, return_tensors="pt", truncation="only_second",
                  max_length=384)
        with torch.no_grad():
            out = qa_model(**enc)
        start = int(out.start_logits.argmax())
        end = int(out.end_logits.argmax())
        if end < start:
            end = start
        ids = enc["input_ids"][0][start:end + 1]
        return tok.decode(ids, skip_special_tokens=True)

    # A pool of distinct contexts and gold answers for "non-equivalent" injection.
    contexts = [r["context"] for r in rows]
    gold_answers = [r["answers"]["text"][0] if r["answers"]["text"] else "" for r in rows]
    shifted = np.roll(np.arange(len(rows)), 1)  # each item paired with a different one

    recs = []
    for i, r in enumerate(rows):
        q = r["question"]
        golds = r["answers"]["text"] or [""]
        wrong_ctx = contexts[shifted[i]]
        # Reader answers under correct and faulted contexts.
        f1_correct = best_f1(answer(q, r["context"]), golds)
        f1_wrong_ctx = best_f1(answer(q, wrong_ctx), golds)
        f1_trunc = best_f1(answer(q, r["context"][: max(1, len(r["context"]) // 4)]), golds)
        # Answer-cache false hit: a different question's gold answer is returned verbatim.
        f1_wrong_ans = best_f1(gold_answers[shifted[i]], golds)
        recs.append(dict(f1_correct=f1_correct, f1_wrong_ctx=f1_wrong_ctx,
                         f1_trunc=f1_trunc, f1_wrong_ans=f1_wrong_ans))
        if (i + 1) % 50 == 0:
            print(f"[downstream] {i + 1}/{len(rows)} questions")

    d = pd.DataFrame(recs)
    # Persist per-question F1 so the figure can bootstrap confidence bands over questions.
    d.to_csv(ROOT / "results" / "downstream_perquestion.csv", index=False, encoding="utf-8")
    base = d.f1_correct.mean()
    out = []
    conditions = {
        "context cache (wrong passage)": d.f1_wrong_ctx.mean(),
        "context cache (truncated passage)": d.f1_trunc.mean(),
        "answer cache (wrong answer)": d.f1_wrong_ans.mean(),
    }
    for name, fault_f1 in conditions.items():
        for p in [0.0, 0.01, 0.02, 0.05, 0.10]:
            f1 = (1 - p) * base + p * fault_f1
            out.append(dict(condition=name, p=p, f1=round(f1, 4),
                            drop_per_point=round(base - fault_f1, 4)))
    res = pd.DataFrame(out)
    (ROOT / "results").mkdir(exist_ok=True)
    res.to_csv(ROOT / "results" / "downstream_summary.csv", index=False, encoding="utf-8")
    print(f"\n[downstream] base F1 (correct context) = {base:.3f} over {len(rows)} questions")
    print(res.to_string(index=False))
    print(f"[downstream] -> results/downstream_summary.csv")


if __name__ == "__main__":
    main()
