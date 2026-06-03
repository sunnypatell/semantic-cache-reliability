# SPDX-License-Identifier: LicenseRef-Proprietary
# Copyright (c) 2026 Sunny Patel. All rights reserved.
"""Encoders that turn prompts into similarity scores.

Every encoder exposes the same interface: ``encode`` maps texts to L2-normalized
vectors, and ``pair_similarity`` maps two aligned lists of texts to a vector of
cosine similarities in [-1, 1]. A lexical TF-IDF encoder is included as a
non-neural control; comparing it against sentence embeddings isolates how much of
the cache's behavior comes from semantics rather than surface word overlap.

Sentence-Transformer model identifiers and revisions are pinned for reproducibility;
see the model cards in the paper appendix.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


class Encoder(Protocol):
    name: str

    def pair_similarity(self, a: Sequence[str], b: Sequence[str]) -> np.ndarray:
        ...


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


@dataclass
class SentenceEncoder:
    """Wrapper around a Sentence-Transformers bi-encoder.

    The heavy ``sentence_transformers`` import is deferred to construction so that
    modules importing :mod:`cacherel.embeddings` for type information do not pay for
    Torch. ``model_name`` is a Hugging Face identifier; ``revision`` pins the weights.
    """

    model_name: str
    revision: str | None = None
    batch_size: int = 256
    device: str | None = None
    prefix: str = ""  # prepended to every text (e.g., "query: " for the E5 models)
    name: str = ""

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = self.name or self.model_name.split("/")[-1]
        self._model = SentenceTransformer(
            self.model_name, revision=self.revision, device=self.device
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        items = [self.prefix + t for t in texts] if self.prefix else list(texts)
        vecs = self._model.encode(
            items,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def pair_similarity(self, a: Sequence[str], b: Sequence[str]) -> np.ndarray:
        if len(a) != len(b):
            raise ValueError("pair_similarity expects aligned, equal-length inputs")
        ea = self.encode(a)
        eb = self.encode(b)
        return np.sum(ea * eb, axis=1).astype(np.float64)


@dataclass
class LexicalEncoder:
    """TF-IDF cosine similarity: a transparent lexical control.

    Fit the vectorizer once on the full corpus of texts the experiment will score,
    then ``pair_similarity`` returns cosine similarity between the TF-IDF vectors of
    aligned pairs. This measures word-overlap similarity with no learned semantics.
    """

    name: str = "tfidf-lexical"
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 1

    def fit(self, corpus: Sequence[str]) -> "LexicalEncoder":
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range, min_df=self.min_df, lowercase=True
        )
        self._vectorizer.fit(list(corpus))
        return self

    def pair_similarity(self, a: Sequence[str], b: Sequence[str]) -> np.ndarray:
        if not hasattr(self, "_vectorizer"):
            # Fit on the union of the two sides if not pre-fit.
            self.fit(list(a) + list(b))
        va = self._vectorizer.transform(list(a))
        vb = self._vectorizer.transform(list(b))
        # Rows are L2-normalized by TfidfVectorizer; cosine is the row-wise dot.
        num = np.asarray(va.multiply(vb).sum(axis=1)).ravel()
        return num.astype(np.float64)


# Pinned encoder roster for the main experiment. Identifiers are Hugging Face hub
# names; revisions are filled when the models are first downloaded and cards written.
DEFAULT_MODEL_SPECS: list[dict[str, str]] = [
    {"model_name": "sentence-transformers/all-MiniLM-L6-v2"},
    {"model_name": "sentence-transformers/all-mpnet-base-v2"},
    {"model_name": "intfloat/e5-small-v2", "prefix": "query: "},
    {"model_name": "intfloat/e5-large-v2", "prefix": "query: "},
    {"model_name": "BAAI/bge-base-en-v1.5"},
    {"model_name": "BAAI/bge-large-en-v1.5"},
]

