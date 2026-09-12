from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="session")
def tool_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a disposable corpus; never populate canonical runtime paths."""
    root = tmp_path_factory.mktemp("tool-corpus")
    staging = root / "data/staging"
    state = root / "state"
    required = {
        staging / "nist_controls_raw.parquet": pd.DataFrame(
            [{"control_id": "AC-2", "family": "AC", "title": "Account Management", "description": "Manage accounts."}]
        ),
        staging / "cfr_raw.parquet": pd.DataFrame(
            [
                {
                    "regulation_id": "12-CFR-1005:policy-v2",
                    "source_type": "eCFR",
                    "source_file": "fixture",
                    "business_valid_from": pd.Timestamp("2024-01-01", tz="UTC"),
                    "text": "Electronic transfer error-resolution requirements.",
                }
            ]
        ),
        staging / "transactions_raw.parquet": pd.DataFrame(
            [
                {
                    "transaction_id": "TX-FIXTURE",
                    "entity_id": "ENTITY-00001",
                    "event_timestamp": pd.Timestamp("2023-01-01", tz="UTC"),
                    "amount": 1.0,
                    "channel": "ACH",
                    "injected_anomaly": False,
                }
            ]
        ),
    }
    for path, frame in required.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    manifest = state / "artifact_manifest.json"
    state.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "path": path.relative_to(root).as_posix(),
                        "sha256": _sha256(path),
                        "bytes": path.stat().st_size,
                        "kind": "test_fixture",
                        "phase": "TEST_BOOTSTRAP",
                    }
                    for path in required
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return root
