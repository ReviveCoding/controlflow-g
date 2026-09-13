from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _core(path: Path, truth: pd.DataFrame) -> np.ndarray:
    output = pd.read_parquet(path).set_index("case_id").loc[truth.index]
    values = (
        output["severity"].eq(truth["truth_severity"])
        & output["disposition"].eq(truth["expected_disposition"])
        & output["action"].eq(truth["expected_action"])
        & output["policy_id"].eq(truth["expected_policy_id"])
        & output.apply(
            lambda row: set(row["evidence_ids"]) == set(truth.loc[row.name, "expected_evidence_ids"]), axis=1
        )
        & output["structured_output_valid"].astype(bool)
    ).to_numpy(dtype=bool)
    return np.asarray(values, dtype=bool)


def paired_ablation_statistics(results: Path, truth_path: Path, output_path: Path, seed: int = 20260912) -> Path:
    truth = pd.read_parquet(truth_path).set_index("case_id")
    full = _core(results / "full_v21_development_traces.parquet", truth)
    rng = np.random.default_rng(seed)
    records = []
    for path in sorted(results.glob("no_*_development_traces.parquet")):
        ablated = _core(path, truth)
        differences = full.astype(int) - ablated.astype(int)
        samples = np.asarray(
            [differences[rng.integers(0, len(differences), len(differences))].mean() for _ in range(2000)]
        )
        full_only = int((full & ~ablated).sum())
        ablated_only = int((~full & ablated).sum())
        records.append(
            {
                "experiment_id": path.stem.removesuffix("_development_traces").upper(),
                "cases": len(full),
                "absolute_core_stc_effect": float(differences.mean()),
                "paired_bootstrap_ci95_low": float(np.quantile(samples, 0.025)),
                "paired_bootstrap_ci95_high": float(np.quantile(samples, 0.975)),
                "mcnemar_full_only": full_only,
                "mcnemar_ablated_only": ablated_only,
            }
        )
    pd.DataFrame(records).to_parquet(output_path, index=False)
    return output_path
