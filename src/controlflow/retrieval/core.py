from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from controlflow.schemas import IdentityContext, TemporalEvidence

TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]+")


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN.findall(text)]


@dataclass(frozen=True)
class SearchHit:
    evidence: TemporalEvidence
    score: float
    rank: int


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[SearchHit]: ...


class KeywordRetriever:
    def __init__(self, corpus: list[TemporalEvidence]) -> None:
        self.corpus = corpus

    def search(self, query: str, k: int) -> list[SearchHit]:
        terms = set(tokenize(query))
        scores = [len(terms.intersection(tokenize(item.text))) for item in self.corpus]
        order = np.argsort(scores)[::-1][:k]
        return [
            SearchHit(self.corpus[index], float(scores[index]), rank + 1)
            for rank, index in enumerate(order)
            if scores[index] > 0
        ]


class BM25Retriever:
    def __init__(self, corpus: list[TemporalEvidence]) -> None:
        self.corpus = corpus
        self.index = BM25Okapi([tokenize(item.text) for item in corpus]) if corpus else None

    def search(self, query: str, k: int) -> list[SearchHit]:
        if self.index is None:
            return []
        scores = self.index.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        return [SearchHit(self.corpus[index], float(scores[index]), rank + 1) for rank, index in enumerate(order)]


class DenseRetriever:
    def __init__(self, corpus: list[TemporalEvidence], embeddings: np.ndarray) -> None:
        self.corpus = corpus
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings = embeddings / np.maximum(norms, 1e-12)

    def search_vector(self, query_embedding: np.ndarray, k: int) -> list[SearchHit]:
        query = query_embedding / max(float(np.linalg.norm(query_embedding)), 1e-12)
        scores = self.embeddings @ query
        order = np.argsort(scores)[::-1][:k]
        return [SearchHit(self.corpus[index], float(scores[index]), rank + 1) for rank, index in enumerate(order)]


def reciprocal_rank_fusion(result_sets: list[list[SearchHit]], k: int, constant: int = 60) -> list[SearchHit]:
    scores: dict[str, float] = {}
    evidence: dict[str, TemporalEvidence] = {}
    for results in result_sets:
        for hit in results:
            evidence[hit.evidence.evidence_id] = hit.evidence
            scores[hit.evidence.evidence_id] = scores.get(hit.evidence.evidence_id, 0.0) + 1.0 / (constant + hit.rank)
    order = sorted(scores, key=scores.get, reverse=True)[:k]  # type: ignore[arg-type]
    return [SearchHit(evidence[item], scores[item], rank + 1) for rank, item in enumerate(order)]


def authorized_evidence_partition(
    corpus: list[TemporalEvidence],
    *,
    identity: IdentityContext,
    event_time: datetime,
    known_time: datetime,
) -> list[TemporalEvidence]:
    """Return the only corpus a downstream scorer is permitted to observe."""
    return [
        evidence
        for evidence in corpus
        if evidence.valid_at(event_time, known_time)
        and identity.clearance >= evidence.classification
        and (not evidence.authorized_roles or identity.role in evidence.authorized_roles)
    ]


class GovernedBM25Retriever(BM25Retriever):
    """BM25 index built exclusively over a point-in-time authorized partition."""

    def __init__(
        self,
        corpus: list[TemporalEvidence],
        *,
        identity: IdentityContext,
        event_time: datetime,
        known_time: datetime,
    ) -> None:
        super().__init__(
            authorized_evidence_partition(
                corpus,
                identity=identity,
                event_time=event_time,
                known_time=known_time,
            )
        )


class NeuralHybridRetriever:
    """Sparse+dense retrieval with a pinned cross-encoder reranker.

    The instance receives an already authorized point-in-time partition, so
    neither dense scoring nor reranking can observe excluded evidence.
    """

    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBED_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
    RERANK_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"

    def __init__(self, corpus: list[TemporalEvidence], *, require_cuda: bool = True) -> None:
        import torch

        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError("neural retrieval refused CPU fallback")
        self.corpus = corpus
        self.sparse = BM25Retriever(corpus)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder, self.reranker = _neural_models(device)
        embeddings = self.embedder.encode(
            [item.text for item in corpus],
            batch_size=128,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.dense = DenseRetriever(corpus, np.asarray(embeddings))

    def search(self, query: str, k: int) -> list[SearchHit]:
        if not self.corpus:
            return []
        query_vector = np.asarray(self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0])
        candidates = reciprocal_rank_fusion(
            [self.sparse.search(query, 40), self.dense.search_vector(query_vector, 40)],
            min(40, len(self.corpus)),
        )
        scores = self.reranker.predict([(query, hit.evidence.text) for hit in candidates])
        order = np.argsort(np.asarray(scores))[::-1][:k]
        return [
            SearchHit(candidates[index].evidence, float(scores[index]), rank + 1) for rank, index in enumerate(order)
        ]


@lru_cache(maxsize=2)
def _neural_models(device: str):  # type: ignore[no-untyped-def]
    from sentence_transformers import CrossEncoder, SentenceTransformer

    return (
        SentenceTransformer(
            NeuralHybridRetriever.EMBED_MODEL,
            revision=NeuralHybridRetriever.EMBED_REVISION,
            device=device,
        ),
        CrossEncoder(
            NeuralHybridRetriever.RERANK_MODEL,
            revision=NeuralHybridRetriever.RERANK_REVISION,
            device=device,
        ),
    )
