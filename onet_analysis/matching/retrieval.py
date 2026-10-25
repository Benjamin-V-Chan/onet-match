"""Retrievers (TF-IDF, BM25, dense bi-encoder) and an optional cross-encoder reranker.

Every retriever indexes one text per occupation (``fit``) and scores queries
against all of them (``score`` -> ``(n_queries, n_occupations)`` array). With
~900 occupations a full score matrix is tiny, so no ANN index is needed.

:class:`Ranker` composes a retriever with an optional reranker and returns the
top-``k`` candidates per query, which is what evaluation and serving consume.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

log = logging.getLogger("onet_analysis.matching.retrieval")

# Query/document prefixes the embedding models were trained with.
MODEL_PROMPTS: dict[str, tuple[str, str]] = {
    "bge": ("Represent this sentence for searching relevant passages: ", ""),
    "e5": ("query: ", "passage: "),
}


def default_prompts(model_name: str) -> tuple[str, str]:
    """Prompts for known base models. Our fine-tuned models (a local dir with a
    ``training.json``) were trained without prompts."""
    if (Path(model_name) / "training.json").is_file():
        return ("", "")
    name = model_name.lower()
    for key, prompts in MODEL_PROMPTS.items():
        if key in name:
            return prompts
    return ("", "")


def best_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:  # pragma: no cover
        pass
    return "cpu"


class Retriever:
    """Interface: ``fit(docs)`` then ``score(queries)``."""

    name = "base"

    def fit(self, docs: Sequence[str]) -> "Retriever":
        raise NotImplementedError

    def score(self, queries: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def config(self) -> dict[str, Any]:
        return {"type": self.name}


class TfidfRetriever(Retriever):
    """The project's existing TF-IDF setup (uni+bigrams, English stop words), cosine."""

    name = "tfidf"

    def __init__(self, **vectorizer_kwargs):
        params = dict(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=1)
        params.update(vectorizer_kwargs)
        self.params = params
        self.vectorizer = TfidfVectorizer(**params)

    def fit(self, docs):
        self.matrix = self.vectorizer.fit_transform(docs)
        return self

    def score(self, queries):
        q = self.vectorizer.transform(queries)
        return np.asarray((q @ self.matrix.T).todense(), dtype=np.float32)


class BM25Retriever(Retriever):
    """Okapi BM25 over unigram tokens, vectorised with sparse matrices."""

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75, stop_words: str | None = "english"):
        self.k1, self.b, self.stop_words = k1, b, stop_words
        self.vectorizer = CountVectorizer(lowercase=True, stop_words=stop_words)

    def fit(self, docs):
        tf = self.vectorizer.fit_transform(docs).tocsr().astype(np.float32)
        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * dl / dl.mean())
        tf = tf.tocoo()
        data = tf.data * (self.k1 + 1) / (tf.data + norm[tf.row]) * self.idf[tf.col]
        self.weights = sparse.csr_matrix((data, (tf.row, tf.col)), shape=tf.shape)
        return self

    def score(self, queries):
        q = self.vectorizer.transform(queries)  # term counts in the query
        return np.asarray((q @ self.weights.T).todense(), dtype=np.float32)

    def config(self):
        return {"type": self.name, "k1": self.k1, "b": self.b}


class DenseRetriever(Retriever):
    """Sentence-transformers bi-encoder; cosine similarity of normalised embeddings."""

    name = "dense"

    def __init__(
        self,
        model_name: str,
        *,
        query_prompt: str | None = None,
        doc_prompt: str | None = None,
        device: str | None = None,
        batch_size: int = 64,
        max_seq_length: int | None = 256,
        model=None,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        qp, dp = default_prompts(model_name)
        self.query_prompt = qp if query_prompt is None else query_prompt
        self.doc_prompt = dp if doc_prompt is None else doc_prompt
        self.device = device or best_device()
        self.batch_size = batch_size
        self.model = model or SentenceTransformer(model_name, device=self.device)
        if max_seq_length:
            self.model.max_seq_length = min(max_seq_length, self.model.max_seq_length or max_seq_length)
        self.doc_embeddings: np.ndarray | None = None

    def encode(self, texts, prompt=""):
        return self.model.encode(
            list(texts), prompt=prompt or None, batch_size=self.batch_size,
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        ).astype(np.float32)

    def fit(self, docs):
        self.doc_embeddings = self.encode(docs, self.doc_prompt)
        return self

    def score(self, queries):
        return self.encode(queries, self.query_prompt) @ self.doc_embeddings.T

    def config(self):
        return {"type": self.name, "model": self.model_name, "query_prompt": self.query_prompt,
                "doc_prompt": self.doc_prompt, "max_seq_length": self.model.max_seq_length}


class CrossEncoderReranker:
    """Rescore (query, occupation document) pairs with a cross-encoder."""

    def __init__(self, model_name: str, *, device: str | None = None, batch_size: int = 64,
                 max_length: int = 320, model=None):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.device = device or best_device()
        self.batch_size = batch_size
        self.model = model or CrossEncoder(model_name, device=self.device, max_length=max_length)

    def rerank(self, queries: Sequence[str], candidates: np.ndarray, docs: Sequence[str]) -> np.ndarray:
        pairs = [(q, docs[j]) for q, row in zip(queries, candidates) for j in row]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False,
                                    convert_to_numpy=True, activation_fn=None)
        return np.asarray(scores, dtype=np.float32).reshape(candidates.shape)

    def config(self):
        return {"type": "cross-encoder", "model": self.model_name}


@dataclass
class Ranking:
    """Top-k candidates per query (indices into the occupation list), best first."""

    indices: np.ndarray  # (n, k) int
    scores: np.ndarray   # (n, k) float — the score used for ranking/calibration


@dataclass
class Ranker:
    """Retriever + optional reranker over a fixed occupation list."""

    retriever: Retriever
    docs: list[str]
    reranker: CrossEncoderReranker | None = None
    k: int = 20
    fuse: float | None = None  # weight of the reranker score in a z-scored blend with the retriever score
    extra: dict = field(default_factory=dict)

    def candidates(self, queries: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Retriever top-k (indices, scores), best first."""
        k = min(self.k, len(self.docs))
        s = self.retriever.score(list(queries))
        top = np.argpartition(-s, kth=k - 1, axis=1)[:, :k]
        top_s = np.take_along_axis(s, top, axis=1)
        order = np.argsort(-top_s, axis=1, kind="stable")
        return np.take_along_axis(top, order, 1), np.take_along_axis(top_s, order, 1)

    def rank(self, queries: Sequence[str], batch_size: int = 512) -> Ranking:
        idx_parts, score_parts = [], []
        for start in range(0, len(queries), batch_size):
            batch = list(queries[start:start + batch_size])
            top, top_s = self.candidates(batch)
            if self.reranker is not None:
                ce = self.reranker.rerank(batch, top, self.docs)
                top_s = ce if self.fuse is None else fuse_scores(top_s, ce, self.fuse)
                order = np.argsort(-top_s, axis=1, kind="stable")
                top, top_s = np.take_along_axis(top, order, 1), np.take_along_axis(top_s, order, 1)
            idx_parts.append(top)
            score_parts.append(top_s)
        return Ranking(np.vstack(idx_parts), np.vstack(score_parts))

    def config(self) -> dict:
        return {"retriever": self.retriever.config(),
                "reranker": self.reranker.config() if self.reranker else None, "k": self.k,
                **({"fuse": self.fuse} if self.fuse is not None else {}), **self.extra}


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)


def fuse_scores(retriever_scores: np.ndarray, reranker_scores: np.ndarray, weight: float) -> np.ndarray:
    return (1 - weight) * _zscore(retriever_scores) + weight * _zscore(reranker_scores)


def softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    z = scores / max(temperature, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def log_space(lo: float, hi: float, n: int) -> np.ndarray:
    return np.exp(np.linspace(math.log(lo), math.log(hi), n))
