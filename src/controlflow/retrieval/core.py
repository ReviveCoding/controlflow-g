from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
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
        self.index = BM25Okapi([tokenize(item.text) for item in corpus])

    def search(self, query: str, k: int) -> list[SearchHit]:
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


def governed_filter(
    hits: list[SearchHit],
    *,
    identity: IdentityContext,
    event_time: datetime,
    known_time: datetime,
    k: int,
) -> list[SearchHit]:
    accepted = [
        hit
        for hit in hits
        if hit.evidence.valid_at(event_time, known_time)
        and identity.clearance >= hit.evidence.classification
        and (not hit.evidence.authorized_roles or identity.role in hit.evidence.authorized_roles)
    ][:k]
    return [SearchHit(hit.evidence, hit.score, rank + 1) for rank, hit in enumerate(accepted)]
