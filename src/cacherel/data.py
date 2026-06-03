# SPDX-License-Identifier: LicenseRef-Proprietary
# Copyright (c) 2026 Sunny Patel. All rights reserved.
"""Labeled prompt-pair datasets that operationalize cache-equivalence.

The central measurement question is not "do two prompts look similar?" but "is it
safe to serve the stored answer for one when the other arrives?" Embedding cosine
answers the first; we need human judgments for the second. We therefore ground the
study in public, human-labeled pair corpora whose labels encode same-meaning or
same-intent rather than surface overlap:

  qqp  : Quora Question Pairs. Two questions, labeled duplicate (1) or not (0).
         A conversational-question domain. Duplicate questions are, by construction,
         questions that should receive the same answer.
  paws : Paraphrase Adversaries from Word Scrambling (labeled_final). Pairs with high
         lexical overlap, labeled paraphrase (1) or not (0). These are the decisive
         hard negatives: prompts that look almost identical yet are not equivalent,
         which is exactly where a similarity cache produces silent false hits.
  mrpc : Microsoft Research Paraphrase Corpus. Sentence pairs from news, labeled
         semantically equivalent (1) or not (0). A more formal register.

Treating question-duplication and paraphrase as proxies for answer-equivalence is a
defensible but imperfect operationalization; the gap is stated plainly in the paper's
construct-validity and limitations sections. Every loader is deterministic given a seed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairSet:
    """An aligned set of prompt pairs with binary cache-equivalence labels."""

    name: str
    a: list[str]
    b: list[str]
    y: np.ndarray  # shape (n,), 1 = cache-equivalent, 0 = not

    def __len__(self) -> int:
        return len(self.a)

    @property
    def base_rate(self) -> float:
        """Fraction of pairs labeled equivalent."""
        return float(np.mean(self.y))

    def corpus(self) -> list[str]:
        """All texts, for fitting a lexical encoder."""
        return list(self.a) + list(self.b)


_SPECS = {
    "qqp": dict(path="glue", config="qqp", split="train",
                fa="question1", fb="question2", flabel="label"),
    "paws": dict(path="paws", config="labeled_final", split="train",
                 fa="sentence1", fb="sentence2", flabel="label"),
    "mrpc": dict(path="glue", config="mrpc", split="train",
                 fa="sentence1", fb="sentence2", flabel="label"),
}


def available_datasets() -> list[str]:
    return list(_SPECS)


def load_pairs(name: str, *, max_pairs: int | None = 4000, seed: int = 0) -> PairSet:
    """Load a labeled pair set, optionally subsampling to ``max_pairs`` with a seed.

    Subsampling preserves the corpus's natural base rate (we do not rebalance here;
    realistic traffic composition is handled separately). Texts are stripped; empty
    or duplicate-identical pairs are dropped so that exact-match cases do not inflate
    the positive class.
    """
    if name not in _SPECS:
        raise KeyError(f"unknown dataset {name!r}; choose from {available_datasets()}")
    from datasets import load_dataset

    spec = _SPECS[name]
    ds = load_dataset(spec["path"], spec["config"], split=spec["split"])

    a_all = [str(x).strip() for x in ds[spec["fa"]]]
    b_all = [str(x).strip() for x in ds[spec["fb"]]]
    y_all = np.asarray(ds[spec["flabel"]], dtype=int)

    keep = [
        i
        for i in range(len(a_all))
        if a_all[i] and b_all[i] and a_all[i].lower() != b_all[i].lower()
    ]
    a_all = [a_all[i] for i in keep]
    b_all = [b_all[i] for i in keep]
    y_all = y_all[keep]

    if max_pairs is not None and len(a_all) > max_pairs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(a_all), size=max_pairs, replace=False)
        idx.sort()
        a_all = [a_all[i] for i in idx]
        b_all = [b_all[i] for i in idx]
        y_all = y_all[idx]

    return PairSet(name=name, a=a_all, b=b_all, y=y_all)
