from __future__ import annotations

import hashlib
import time

import numpy as np
import pandas as pd

from controlflow.core.state import PhaseRun, ProjectPaths, canonical_json, utc_now
from controlflow.retrieval.core import BM25Retriever
from controlflow.retrieval.experiments import _item


def run() -> str:
    paths = ProjectPaths.discover()
    rows = []
    query = "privileged account access review"
    with PhaseRun("P21", paths) as phase:
        for scale in (10_000, 50_000, 100_000):
            corpus = [
                _item(f"D-{index}", f"financial filing distractor section {index % 1000} revenue controls")
                for index in range(scale - 1)
            ]
            corpus.append(_item("RELEVANT", "privileged account access must be periodically reviewed and disabled"))
            started = time.perf_counter()
            index = BM25Retriever(corpus)
            build = time.perf_counter() - started
            latencies = []
            identifiers = []
            for _ in range(25):
                started = time.perf_counter()
                hits = index.search(query, 10)
                latencies.append((time.perf_counter() - started) * 1000)
                identifiers = [hit.evidence.evidence_id for hit in hits]
            rows.append(
                {
                    "experiment_id": f"retrieval-scale-{scale}",
                    "config_hash": hashlib.sha256(canonical_json({"scale": scale, "engine": "BM25"})).hexdigest(),
                    "dataset_hash": "deterministic-distractors-v1",
                    "split_identifier": "scale",
                    "seed": 0,
                    "hardware_runtime": "CPU",
                    "timestamp": utc_now(),
                    "status": "ok",
                    "scale_chunks": scale,
                    "index_build_seconds": build,
                    "index_size_bytes_proxy": sum(len(item.text.encode()) for item in corpus),
                    "recall_at_10": float("RELEVANT" in identifiers),
                    "p50_latency_ms": float(np.quantile(latencies, 0.5)),
                    "p95_latency_ms": float(np.quantile(latencies, 0.95)),
                    "largest_levels_not_run": "500K/1M omitted under SMOKE RAM reserve",
                }
            )
        target = paths.root / "results/retrieval_scaling.parquet"
        pd.DataFrame(rows).to_parquet(target, index=False)
        phase.register(target, "result_table")
    return str(target)


if __name__ == "__main__":
    print(run())
