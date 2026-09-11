from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from controlflow.core.state import PhaseRun, ProjectPaths, canonical_json, utc_now
from controlflow.eval.metrics import holm_adjust, paired_bootstrap


def run() -> str:
    paths = ProjectPaths.discover()
    traces = pd.read_parquet(paths.root / "results" / "ablation_traces.parquet")
    full = (
        traces[traces.experiment_id == "ablation-AB12_full_controlflow_g"]
        .set_index("case_id")
        .safe_task_completion.astype(float)
    )
    rows = []
    with PhaseRun("P23", paths) as phase:
        for experiment_id, candidate in traces.groupby("experiment_id"):
            if experiment_id == "ablation-AB12_full_controlflow_g":
                continue
            paired = candidate.set_index("case_id").loc[full.index].safe_task_completion.astype(float)
            ci = paired_bootstrap(full.to_numpy(), paired.to_numpy(), repeats=2000, seed=1729)
            b = int(((full == 1) & (paired == 0)).sum())
            c = int(((full == 0) & (paired == 1)).sum())
            p = float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0
            rows.append(
                {
                    "experiment_id": f"stats-{experiment_id}",
                    "config_hash": hashlib.sha256(
                        canonical_json({"comparison": experiment_id, "repeats": 2000})
                    ).hexdigest(),
                    "dataset_hash": "ablation-traces",
                    "split_identifier": "validation",
                    "seed": 1729,
                    "hardware_runtime": "CPU",
                    "timestamp": utc_now(),
                    "status": "ok",
                    "comparison": experiment_id,
                    "effect_stc_absolute": ci["effect"],
                    "ci95_low": ci["ci95_low"],
                    "ci95_high": ci["ci95_high"],
                    "mcnemar_discordant_full_wins": b,
                    "mcnemar_discordant_candidate_wins": c,
                    "p_value": p,
                }
            )
        adjusted = holm_adjust([row["p_value"] for row in rows])
        for row, value in zip(rows, adjusted, strict=True):
            row["holm_adjusted_p"] = value
        cells = {
            "verifier_x_authorization": (
                "ablation-AB06_no_verifier",
                "ablation-AB07_no_authorization",
                "ablation-INT_no_verifier_no_authorization",
            ),
            "calibration_x_hitl": (
                "ablation-AB04_no_calibration",
                "ablation-AB08_no_hitl",
                "ablation-INT_no_calibration_no_hitl",
            ),
            "temporal_x_verifier": (
                "ablation-AB01_no_temporal_retrieval",
                "ablation-AB06_no_verifier",
                "ablation-INT_no_temporal_no_verifier",
            ),
        }
        rng = np.random.default_rng(1729)
        indexed = {
            name: group.set_index("case_id").loc[full.index].safe_task_completion.astype(float)
            for name, group in traces.groupby("experiment_id")
        }
        for name, (without_a, without_b, without_both) in cells.items():
            contrast = full - indexed[without_a] - indexed[without_b] + indexed[without_both]
            samples = np.asarray(
                [contrast.iloc[rng.integers(0, len(contrast), len(contrast))].mean() for _ in range(2_000)]
            )
            p_value = min(1.0, 2 * min(float((samples <= 0).mean()), float((samples >= 0).mean())))
            rows.append(
                {
                    "experiment_id": f"stats-interaction-{name}",
                    "config_hash": hashlib.sha256(canonical_json({"interaction": name})).hexdigest(),
                    "dataset_hash": "ablation-traces",
                    "split_identifier": "validation",
                    "seed": 1729,
                    "hardware_runtime": "CPU",
                    "timestamp": utc_now(),
                    "status": "ok",
                    "comparison": f"interaction-{name}",
                    "effect_stc_absolute": float(contrast.mean()),
                    "ci95_low": float(np.quantile(samples, 0.025)),
                    "ci95_high": float(np.quantile(samples, 0.975)),
                    "mcnemar_discordant_full_wins": 0,
                    "mcnemar_discordant_candidate_wins": 0,
                    "p_value": p_value,
                    "holm_adjusted_p": float("nan"),
                }
            )
        target = paths.root / "results" / "validation_statistics.parquet"
        pd.DataFrame(rows).to_parquet(target, index=False)
        phase.register(target, "result_table")
    return str(target)


if __name__ == "__main__":
    print(run())
