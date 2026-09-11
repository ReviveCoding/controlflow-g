from __future__ import annotations

import hashlib

import pandas as pd

from controlflow.core.state import PhaseRun, ProjectPaths, canonical_json, utc_now
from controlflow.data.contracts import ColumnRule, DataContractError, QualityAction, SourceClass, validate_frame

SCENARIOS = {
    "duplicate event": pd.DataFrame({"id": ["a", "a"], "timestamp": ["2025-01-01", "2025-01-01"], "value": [1, 1]}),
    "missing key": pd.DataFrame({"timestamp": ["2025-01-01"], "value": [1]}),
    "new column": pd.DataFrame({"id": ["a"], "timestamp": ["2025-01-01"], "value": [1], "new": [2]}),
    "type change": pd.DataFrame({"id": ["a", "b"], "timestamp": ["2025-01-01", "2025-01-02"], "value": [1, "bad"]}),
    "corrupt row": pd.DataFrame({"id": ["a", "b"], "timestamp": ["2025-01-01", "not-a-date"], "value": [1, 2]}),
}
RULES = (ColumnRule("id", "string"), ColumnRule("timestamp", "datetime64[ns, UTC]"), ColumnRule("value", "int64"))


def run() -> str:
    paths = ProjectPaths.discover()
    rows = []
    with PhaseRun("P20", paths) as phase:
        for name, frame in SCENARIOS.items():
            detected = False
            quarantined = 0
            try:
                result = validate_frame(
                    frame,
                    RULES,
                    source_class=SourceClass.STRICT,
                    invalid_action=QualityAction.QUARANTINE,
                    primary_key="id",
                )
                detected = result.metrics["invalid_rows"] > 0
                quarantined = result.metrics["rows_quarantined"]
            except DataContractError:
                detected = True
            rows.append(
                {
                    "experiment_id": f"data-failure-{name.replace(' ', '-')}",
                    "config_hash": hashlib.sha256(canonical_json({"scenario": name})).hexdigest(),
                    "dataset_hash": "data-failure-fixtures-v1",
                    "split_identifier": "failure_validation",
                    "seed": 0,
                    "hardware_runtime": "CPU",
                    "timestamp": utc_now(),
                    "status": "ok",
                    "scenario": name,
                    "injected": True,
                    "detected": detected,
                    "quarantined_rows": quarantined,
                }
            )
        for name in (
            "late event",
            "out-of-order event",
            "event replay",
            "schema drift",
            "source outage",
            "stream restart",
            "broken document",
        ):
            # These have dedicated executed evidence in P06 or parser failure
            # paths; link it rather than claiming a new injection.
            rows.append(
                {
                    "experiment_id": f"data-failure-{name.replace(' ', '-')}",
                    "config_hash": hashlib.sha256(canonical_json({"scenario": name})).hexdigest(),
                    "dataset_hash": "linked-P06-or-parser",
                    "split_identifier": "failure_validation",
                    "seed": 0,
                    "hardware_runtime": "CPU",
                    "timestamp": utc_now(),
                    "status": "linked_evidence",
                    "scenario": name,
                    "injected": False,
                    "detected": False,
                    "quarantined_rows": 0,
                }
            )
        target = paths.root / "results/data_failures.parquet"
        pd.DataFrame(rows).to_parquet(target, index=False)
        phase.register(target, "result_table")
    return str(target)


if __name__ == "__main__":
    print(run())
