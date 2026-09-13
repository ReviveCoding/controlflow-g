from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict

from controlflow.core.state import atomic_write_json, sha256_file, utc_now


class BundleArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    artifact_id: str
    path: str
    sha256: str
    training_dataset_hash: str
    calibration_dataset_hash: str
    validation_dataset_hash: str
    selection_rule: str
    selected_model_id: str
    seed: int


def verify_candidate_bundle(path: Path, root: Path) -> dict[str, Any]:
    bundle = json.loads(path.read_text(encoding="utf-8"))
    for item in bundle["artifacts"]:
        artifact = (root / item["path"]).resolve()
        artifact.relative_to(root.resolve())
        if not artifact.is_file() or sha256_file(artifact) != item["sha256"]:
            raise RuntimeError(f"CANDIDATE_BUNDLE_MISMATCH: {item['artifact_id']}")
    return cast(dict[str, Any], bundle)


def write_bundle(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, {"schema_version": 1, "created_at": utc_now(), **payload})
