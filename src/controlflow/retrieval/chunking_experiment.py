from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from typing import Any

import pandas as pd

from controlflow.core.state import PhaseRun, ProjectPaths, canonical_json, sha256_file, utc_now
from controlflow.retrieval.chunking import Chunk, control_chunks, fixed_chunks, hierarchical_chunks, section_chunks
from controlflow.retrieval.core import BM25Retriever
from controlflow.retrieval.evaluate import RetrievalCase, evaluate_retriever
from controlflow.retrieval.experiments import QUERIES, _item


def run() -> str:
    paths = ProjectPaths.discover()
    source = paths.root / "data/staging/nist_controls_raw.parquet"
    controls = pd.read_parquet(source)
    document = "\n".join(
        f"{row.control_id} {str(row.title).upper()}\n{row.description}" for row in controls.itertuples()
    )
    strategies: dict[str, Callable[[], list[Chunk]]] = {
        "fixed_256": lambda: fixed_chunks("nist", document, 256),
        "fixed_512": lambda: fixed_chunks("nist", document, 512),
        "fixed_1024": lambda: fixed_chunks("nist", document, 1024),
        "section_aware": lambda: section_chunks("nist", document),
        "control_aware": lambda: control_chunks("nist", document),
        "hierarchical": lambda: hierarchical_chunks("nist", document),
    }
    rows: list[dict[str, Any]] = []
    with PhaseRun("P14", paths) as phase:
        for name, builder in strategies.items():
            started = time.perf_counter()
            chunks = builder()
            evidence = [_item(chunk.chunk_id, chunk.text) for chunk in chunks]
            retriever = BM25Retriever(evidence)
            build_seconds = time.perf_counter() - started
            cases = [
                RetrievalCase(
                    f"chunk-{control_id}",
                    query,
                    frozenset(chunk.chunk_id for chunk in chunks if control_id in chunk.text),
                )
                for control_id, query in QUERIES.items()
            ]
            metrics = evaluate_retriever(retriever, cases)
            rows.append(
                {
                    "experiment_id": f"chunking-{name}",
                    "config_hash": hashlib.sha256(canonical_json({"strategy": name})).hexdigest(),
                    "dataset_hash": sha256_file(source),
                    "split_identifier": "independent_queries_v2",
                    "seed": 0,
                    "hardware_runtime": "CPU",
                    "timestamp": utc_now(),
                    "status": "ok",
                    "strategy": name,
                    "chunk_count": len(chunks),
                    "index_build_seconds": build_seconds,
                    "index_size_bytes_proxy": sum(len(chunk.text.encode()) for chunk in chunks),
                    **metrics,
                }
            )
        target = paths.root / "results/chunking.parquet"
        pd.DataFrame(rows).to_parquet(target, index=False)
        phase.register(target, "result_table")
    return str(target)


if __name__ == "__main__":
    print(run())
