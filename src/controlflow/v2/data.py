from __future__ import annotations

import hashlib
import json
from pathlib import Path

from controlflow.core.state import ProjectPaths, atomic_write_json, sha256_file, utc_now
from controlflow.data.synthetic import generate_cases
from controlflow.features.point_in_time import attach_synthetic_pit_features
from controlflow.v2.schemas import RootCauseCode

ROOT_CAUSE_TRUTH = {
    "normal": RootCauseCode.ROUTINE_VARIANCE,
    "difficult": RootCauseCode.OWNERSHIP_AMBIGUITY,
    "critical": RootCauseCode.MATERIAL_CONTROL_BREAKDOWN,
    "rare": RootCauseCode.NOVEL_THIRD_PARTY_FAILURE,
    "missing_evidence": RootCauseCode.SOURCE_EVIDENCE_MISSING,
    "conflicting_evidence": RootCauseCode.AUTHORITATIVE_SOURCE_CONFLICT,
    "stale_policy": RootCauseCode.TEMPORAL_POLICY_MISMATCH,
    "privilege": RootCauseCode.AUTHORIZATION_SCOPE_VIOLATION,
    "adversarial": RootCauseCode.PROMPT_INJECTION_ATTEMPT,
}


def _ids_hash(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def create_development_dataset(count: int = 2_000, seed: int = 20_260_912) -> Path:
    """Create V2 development data only. This function cannot create a final holdout."""
    paths = ProjectPaths.discover()
    target = paths.root / "data/v2/development/cases.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_cases(count, seed, case_prefix="V2DEV", entity_prefix="V2DEV-ENTITY", holdout=False)
    frame["root_cause_code"] = frame["case_type"].map(lambda value: ROOT_CAUSE_TRUTH[str(value)].value)
    frame = attach_synthetic_pit_features(frame)
    temporary = target.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(target)

    ordinary = frame.loc[~frame["is_ood"] & ~frame["is_adversarial"]].copy()
    splits = {
        "train": ordinary.loc[ordinary["template_family"].isin([0, 1, 2, 3]), "case_id"].astype(str).tolist(),
        "calibration": ordinary.loc[ordinary["template_family"].eq(4), "case_id"].astype(str).tolist(),
        "validation": ordinary.loc[ordinary["template_family"].isin([5, 6, 7]), "case_id"].astype(str).tolist(),
        "ood": frame.loc[frame["is_ood"], "case_id"].astype(str).tolist(),
        "adversarial": frame.loc[frame["is_adversarial"], "case_id"].astype(str).tolist(),
    }
    membership = [identifier for values in splits.values() for identifier in values]
    if len(membership) != len(set(membership)):
        raise RuntimeError("V2 development split overlap")
    split_dir = paths.root / "data/v2/splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_records = []
    for name, ids in splits.items():
        split_path = split_dir / f"{name}.ids"
        split_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
        split_records.append({"name": name, "count": len(ids), "sha256": _ids_hash(ids)})
    atomic_write_json(
        paths.state / "v2_split_manifest.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "dataset_role": "development_and_validation_only",
            "generator": "controlflow.data.synthetic.generate_cases",
            "generator_version": 5,
            "seed": seed,
            "case_namespace": "V2DEV-*",
            "rows": len(frame),
            "dataset_sha256": sha256_file(target),
            "splits": split_records,
            "fresh_final_holdout_created": False,
            "v1_final_artifacts_read": False,
        },
    )
    audit = {
        "allowed_model_features": [
            "amount",
            "repeat_count",
            "historical_failures",
            "data_sensitivity",
            "business_unit",
            "policy_version",
            "narrative",
            "control_ids",
            "regulation_ids",
            "evidence_status",
            "requested_scope",
        ],
        "prohibited_generator_or_truth_features": [
            "case_type",
            "severity",
            "root_cause_code",
            "expected_disposition",
            "expected_actions",
            "authorization_outcome",
            "review_requirement",
            "is_ood",
            "is_adversarial",
            "future_failures",
        ],
    }
    (paths.state / "v2_feature_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return target


def load_v2_split(name: str) -> list[str]:
    if "final" in name.casefold() or "test" in name.casefold():
        raise PermissionError("V2 optimization code cannot access or create final/test splits")
    path = ProjectPaths.discover().root / f"data/v2/splits/{name}.ids"
    return path.read_text(encoding="utf-8").splitlines()
