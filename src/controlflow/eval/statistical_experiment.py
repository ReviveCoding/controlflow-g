from __future__ import annotations

import hashlib

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
        target = paths.root / "results" / "validation_statistics.parquet"
        pd.DataFrame(rows).to_parquet(target, index=False)
        phase.register(target, "result_table")
    return str(target)


if __name__ == "__main__":
    print(run())
