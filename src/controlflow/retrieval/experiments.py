from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from controlflow.core.resources import GpuSemaphore
from controlflow.core.state import PhaseRun, ProjectPaths, canonical_json, sha256_file, utc_now
from controlflow.retrieval.core import (
    BM25Retriever,
    DenseRetriever,
    KeywordRetriever,
    SearchHit,
    reciprocal_rank_fusion,
)
from controlflow.retrieval.evaluate import RetrievalCase, evaluate_retriever
from controlflow.schemas import TemporalEvidence

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANK_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
QUERIES = {
    "AC-2": "How should employee and service accounts be opened, disabled, and periodically reviewed?",
    "AC-3": "What mechanism enforces approved access decisions for protected systems?",
    "AC-6": "How should organizations minimize administrator privileges?",
    "AU-2": "Which security events should the organization capture in its audit records?",
    "AU-6": "How should teams review audit logs and report suspicious activity?",
    "CA-7": "What is expected for continuous monitoring of security controls?",
    "CM-2": "How should an approved baseline configuration be maintained?",
    "IA-2": "How must organizational users prove identity before system access?",
    "IR-4": "What process contains, eradicates, and recovers from security incidents?",
    "RA-5": "How should technical vulnerabilities be scanned and remediated?",
    "SC-7": "How should boundaries between networks and external systems be protected?",
    "SI-4": "How should systems be monitored for attacks and indicators of compromise?",
}


class Adapter:
    def __init__(self, function: Callable[[str, int], list[SearchHit]]):
        self.function = function

    def search(self, query: str, k: int) -> list[SearchHit]:
        return self.function(query, k)


def _item(
    identifier: str, text: str, source: str = "NIST", classification: int = 0, start: int = 2020, end: int | None = None
) -> TemporalEvidence:
    valid_from = datetime(start, 1, 1, tzinfo=UTC)
    valid_to = datetime(end, 1, 1, tzinfo=UTC) if end else None
    return TemporalEvidence(
        evidence_id=identifier,
        source=source,
        text=text,
        classification=classification,
        business_valid_from=valid_from,
        business_valid_to=valid_to,
        system_known_from=valid_from,
        authorized_roles=frozenset({"Control Analyst"}),
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


def _build(paths: ProjectPaths) -> tuple[list[TemporalEvidence], list[RetrievalCase]]:
    controls = pd.read_parquet(paths.root / "data/staging/nist_controls_raw.parquet")
    corpus = [_item(str(row.control_id), f"{row.title}. {row.description}") for row in controls.itertuples()]
    sec = pd.read_parquet(paths.root / "data/staging/sec_filings_raw.parquet")
    corpus.extend(
        _item(f"SEC-{index}", str(row.text)[:800], source="SEC") for index, row in enumerate(sec.itertuples())
    )
    for identifier, query in QUERIES.items():
        corpus.append(_item(f"{identifier}:stale", query + " superseded version", start=2020, end=2024))
        corpus.append(_item(f"{identifier}:restricted", query + " confidential appendix", classification=5))
    cases = [RetrievalCase(f"RQ-{identifier}", query, frozenset({identifier})) for identifier, query in QUERIES.items()]
    return corpus, cases


def _safety(retriever: Adapter | KeywordRetriever | BM25Retriever, cases: list[RetrievalCase]) -> dict[str, float]:
    base = evaluate_retriever(retriever, cases)
    hits = [hit for case in cases for hit in retriever.search(case.query, 10)]
    base.update(
        {
            "evidence_coverage": base["recall_at_10"],
            "citation_precision": base["precision_at_k"],
            "stale_evidence_rate": float(np.mean([hit.evidence.evidence_id.endswith(":stale") for hit in hits]))
            if hits
            else 0.0,
            "unauthorized_retrieval_rate": float(
                np.mean([hit.evidence.evidence_id.endswith(":restricted") for hit in hits])
            )
            if hits
            else 0.0,
        }
    )
    base["temporal_correctness"] = 1.0 - base["stale_evidence_rate"]
    return base


def _record(name: str, metrics: dict[str, float], source_hash: str, runtime: str) -> dict[str, object]:
    return {
        "experiment_id": f"retrieval-{name}",
        "config_hash": hashlib.sha256(
            canonical_json({"name": name, "embed": EMBED_REVISION, "rerank": RERANK_REVISION})
        ).hexdigest(),
        "dataset_hash": source_hash,
        "split_identifier": "independent_queries_v2",
        "seed": 17,
        "hardware_runtime": runtime,
        "timestamp": utc_now(),
        "status": "ok",
        "metrics": json.dumps(metrics, sort_keys=True),
    }


def run() -> str:
    paths = ProjectPaths.discover()
    source = paths.root / "data/staging/nist_controls_raw.parquet"
    corpus, cases = _build(paths)
    query_map = {case.query: index for index, case in enumerate(cases)}
    if not torch.cuda.is_available():
        raise RuntimeError("dense retrieval refused CPU fallback")
    with GpuSemaphore():
        encoder = SentenceTransformer(
            EMBED_MODEL, revision=EMBED_REVISION, device="cuda", model_kwargs={"torch_dtype": torch.float16}
        )
        vectors = encoder.encode(
            [item.text for item in corpus],
            batch_size=128,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_vectors = encoder.encode(
            [case.query for case in cases],
            batch_size=128,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        device = str(next(encoder.parameters()).device)
        del encoder
        torch.cuda.empty_cache()
    if not device.startswith("cuda"):
        raise RuntimeError("embedding CPU fallback")

    def make_retrievers(subset: list[int]) -> tuple[BM25Retriever, Adapter, Adapter]:
        items = [corpus[index] for index in subset]
        bm25 = BM25Retriever(items)
        dense = DenseRetriever(items, vectors[subset])
        dense_adapter = Adapter(lambda query, k: dense.search_vector(query_vectors[query_map[query]], k))
        hybrid = Adapter(
            lambda query, k: reciprocal_rank_fusion(
                [bm25.search(query, max(k, 20)), dense_adapter.search(query, max(k, 20))], k
            )
        )
        return bm25, dense_adapter, hybrid

    all_indices = list(range(len(corpus)))
    bm25, dense, hybrid = make_retrievers(all_indices)
    keyword = KeywordRetriever(corpus)
    with PhaseRun("P14", paths) as phase:
        records = [
            _record("R0_keyword", _safety(keyword, cases), sha256_file(source), "CPU"),
            _record("R1_bm25", _safety(bm25, cases), sha256_file(source), "CPU"),
            _record("R2_dense", _safety(dense, cases), sha256_file(source), f"CUDA:{device}"),
            _record("R3_hybrid", _safety(hybrid, cases), sha256_file(source), "CPU+CUDA"),
        ]
        target = paths.root / "results/retrieval.parquet"
        pd.DataFrame(records).to_parquet(target, index=False)
        phase.register(target, "result_table")
    event = datetime(2025, 1, 1, tzinfo=UTC)
    subsets = {
        "R4_hybrid_reranker": all_indices,
        "R5_metadata": [i for i, x in enumerate(corpus) if x.source == "NIST"],
        "R6_temporal": [i for i, x in enumerate(corpus) if x.source == "NIST" and x.valid_at(event, event)],
        "R7_authorized": [
            i
            for i, x in enumerate(corpus)
            if x.source == "NIST"
            and x.valid_at(event, event)
            and x.classification <= 2
            and "Control Analyst" in x.authorized_roles
        ],
    }
    advanced = []
    with PhaseRun("P15", paths) as phase, GpuSemaphore():
        reranker = CrossEncoder(
            RERANK_MODEL, revision=RERANK_REVISION, device="cuda", model_kwargs={"torch_dtype": torch.float16}
        )
        for name, indices in subsets.items():
            _, _, candidate = make_retrievers(indices)

            def search(
                query: str,
                k: int,
                candidate: Adapter = candidate,
                model: CrossEncoder = reranker,
            ) -> list[SearchHit]:
                hits = candidate.search(query, 20)
                scores = model.predict(
                    [(query, hit.evidence.text) for hit in hits], batch_size=64, convert_to_numpy=True
                )
                order = np.argsort(scores)[::-1][:k]
                return [
                    SearchHit(hits[index].evidence, float(scores[index]), rank + 1) for rank, index in enumerate(order)
                ]

            advanced.append(
                _record(
                    name,
                    _safety(Adapter(search), cases),
                    sha256_file(source),
                    f"CPU+CUDA:{next(reranker.model.parameters()).device}",
                )
            )
        del reranker
        torch.cuda.empty_cache()
        pd.concat([pd.read_parquet(target), pd.DataFrame(advanced)], ignore_index=True).to_parquet(target, index=False)
        phase.register(target, "result_table")
    return str(target)


if __name__ == "__main__":
    print(run())
