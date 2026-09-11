from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from controlflow.retrieval.core import Retriever


@dataclass(frozen=True)
class RetrievalCase:
    query_id: str
    query: str
    relevant_ids: frozenset[str]


def evaluate_retriever(retriever: Retriever, cases: list[RetrievalCase], k: int = 10) -> dict[str, float]:
    recalls: dict[int, list[float]] = {1: [], 5: [], 10: []}
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    precisions: list[float] = []
    latencies: list[float] = []
    for case in cases:
        started = perf_counter()
        hits = retriever.search(case.query, k)
        latencies.append(perf_counter() - started)
        identifiers = [hit.evidence.evidence_id for hit in hits]
        for cutoff in recalls:
            recalls[cutoff].append(len(set(identifiers[:cutoff]) & case.relevant_ids) / max(1, len(case.relevant_ids)))
        relevant_ranks = [index + 1 for index, identifier in enumerate(identifiers) if identifier in case.relevant_ids]
        reciprocal_ranks.append(1.0 / min(relevant_ranks) if relevant_ranks else 0.0)
        gains = np.asarray([1.0 if identifier in case.relevant_ids else 0.0 for identifier in identifiers[:10]])
        discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
        dcg = float((gains * discounts).sum())
        ideal_count = min(len(case.relevant_ids), 10)
        idcg = float(discounts[:ideal_count].sum()) if ideal_count else 1.0
        ndcgs.append(dcg / idcg)
        precisions.append(len(set(identifiers[:k]) & case.relevant_ids) / max(1, k))
    latency_ms = np.asarray(latencies) * 1000
    return {
        "recall_at_1": float(np.mean(recalls[1])),
        "recall_at_5": float(np.mean(recalls[5])),
        "recall_at_10": float(np.mean(recalls[10])),
        "precision_at_k": float(np.mean(precisions)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "ndcg_at_10": float(np.mean(ndcgs)),
        "p50_latency_ms": float(np.quantile(latency_ms, 0.5)),
        "p95_latency_ms": float(np.quantile(latency_ms, 0.95)),
    }
