from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from controlflow.core.state import PhaseRun, ProjectPaths, sha256_file, utc_now


def run(seed: int = 17, repeats: int = 10_000) -> str:
    paths = ProjectPaths.discover()
    trace_path = paths.root / "results/final_test_traces.parquet"
    traces = pd.read_parquet(trace_path)
    pivot = traces.pivot(index="case_id", columns="experiment_id", values="safe_task_completion").astype(bool)
    baseline = pivot["agent-AG0_rules_templates_final"].to_numpy()
    candidate = pivot["agent-AG6_controlflow_g_final"].to_numpy()
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(pivot), size=(repeats, len(pivot)))
    differences = candidate[indices].mean(axis=1) - baseline[indices].mean(axis=1)
    discordant_candidate = int((candidate & ~baseline).sum())
    discordant_baseline = int((baseline & ~candidate).sum())
    discordant = discordant_candidate + discordant_baseline
    pvalue = float(binomtest(discordant_candidate, discordant, 0.5).pvalue) if discordant else 1.0
    metrics = {
        "sample_size": len(pivot),
        "baseline_stc": float(baseline.mean()),
        "candidate_stc": float(candidate.mean()),
        "paired_stc_difference": float(candidate.mean() - baseline.mean()),
        "paired_bootstrap_ci95_low": float(np.quantile(differences, 0.025)),
        "paired_bootstrap_ci95_high": float(np.quantile(differences, 0.975)),
        "mcnemar_exact_pvalue": pvalue,
        "candidate_only_successes": discordant_candidate,
        "baseline_only_successes": discordant_baseline,
    }
    row = {
        "experiment_id": "final-paired-AG6-vs-AG0",
        "config_hash": sha256_file(paths.root / "state/freeze_manifest.json"),
        "dataset_hash": sha256_file(trace_path),
        "split_identifier": "locked_final_test",
        "seed": seed,
        "hardware_runtime": "CPU",
        "timestamp": utc_now(),
        "status": "ok",
        "metrics": json.dumps(metrics, sort_keys=True),
    }
    target = paths.root / "results/final_statistics.parquet"
    with PhaseRun("P28", paths) as phase:
        pd.DataFrame([row]).to_parquet(target, index=False)
        phase.register(target, "statistical_result")
    return str(target)


if __name__ == "__main__":
    print(run())
