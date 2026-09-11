from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from controlflow.core.state import PhaseRun, ProjectPaths, canonical_json, sha256_file, utc_now
from controlflow.data.splits import load_split


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    ref = np.histogram(reference, bins=edges)[0] / len(reference)
    cur = np.histogram(current, bins=edges)[0] / len(current)
    ref = np.clip(ref, 1e-6, None)
    cur = np.clip(cur, 1e-6, None)
    return float(((cur - ref) * np.log(cur / ref)).sum())


def run() -> str:
    paths = ProjectPaths.discover()
    source = paths.root / "data/silver/synthetic_cases_development.parquet"
    frame = pd.read_parquet(source)
    earlier = frame[frame.case_id.isin(set(load_split("train", phase="P21")))]
    later = frame[frame.case_id.isin(set(load_split("validation", phase="P21")))]
    rows = []
    with PhaseRun("P21", paths) as phase:
        for feature in ("amount", "repeat_count", "historical_failures"):
            rows.append(
                {
                    "signal": feature,
                    "psi": _psi(earlier[feature].to_numpy(), later[feature].to_numpy()),
                    "ks_statistic": float(ks_2samp(earlier[feature], later[feature]).statistic),
                }
            )
        for feature in ("case_type", "severity"):
            categories = sorted(set(earlier[feature]) | set(later[feature]))
            p = np.array([(earlier[feature] == x).mean() for x in categories])
            q = np.array([(later[feature] == x).mean() for x in categories])
            rows.append({"signal": feature, "js_divergence": float(jensenshannon(p, q) ** 2)})
        result = pd.DataFrame(rows)
        result["experiment_id"] = result.signal.map(lambda x: f"drift-{x}")
        result["config_hash"] = result.signal.map(lambda x: hashlib.sha256(canonical_json({"signal": x})).hexdigest())
        result["dataset_hash"] = sha256_file(source)
        result["split_identifier"] = "train_vs_validation"
        result["seed"] = 0
        result["hardware_runtime"] = "CPU"
        result["timestamp"] = utc_now()
        result["status"] = "ok"
        target = paths.root / "results/drift.parquet"
        result.to_parquet(target, index=False)
        phase.register(target, "result_table")
    return str(target)


if __name__ == "__main__":
    print(run())
