from __future__ import annotations

import hashlib
import os
from typing import Any

import pandas as pd

from controlflow.core.state import (
    PhaseRun,
    ProjectPaths,
    atomic_write_json,
    canonical_json,
    sha256_file,
    utc_now,
)


class SealedTestAccessError(PermissionError):
    pass


def create_splits(frame: pd.DataFrame) -> dict[str, list[str]]:
    ordered = frame.sort_values(["event_timestamp", "case_id"]).reset_index(drop=True)
    adversarial = ordered.loc[ordered["is_adversarial"], "case_id"].tolist()
    ood = ordered.loc[ordered["is_ood"] & ~ordered["is_adversarial"], "case_id"].tolist()
    eligible = ordered.loc[~ordered["is_adversarial"] & ~ordered["is_ood"]].copy()
    return {
        "train": eligible.loc[eligible.template_family.isin([0, 1, 2, 3]), "case_id"].tolist(),
        "validation": eligible.loc[eligible.template_family.eq(4), "case_id"].tolist(),
        "temporal_test": eligible.loc[eligible.template_family.eq(5), "case_id"].tolist(),
        "locked_final_test": eligible.loc[eligible.template_family.eq(6), "case_id"].tolist(),
        "entity_disjoint": eligible.loc[eligible.template_family.eq(7), "case_id"].tolist(),
        "ood": ood,
        "adversarial": adversarial,
        "stratified_sanity": ordered.groupby("severity", group_keys=False).head(10)["case_id"].tolist(),
    }


def write_splits(frame: pd.DataFrame) -> None:
    paths = ProjectPaths.discover()
    splits = create_splits(frame)
    split_dir = paths.root / "data" / "splits"
    sealed_dir = paths.root / "data" / "sealed"
    split_dir.mkdir(parents=True, exist_ok=True)
    sealed_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for name, ids in splits.items():
        payload = "\n".join(ids) + "\n"
        target = (sealed_dir if name == "locked_final_test" else split_dir) / f"{name}.ids"
        target.write_text(payload, encoding="utf-8")
        records.append(
            {
                "name": name,
                "count": len(ids),
                "sha256": sha256_file(target),
                "path": target.relative_to(paths.root).as_posix(),
            }
        )
    membership: dict[str, str] = {}
    for name, ids in splits.items():
        for case_id in ids:
            if case_id in membership and name not in {"stratified_sanity"}:
                raise ValueError(f"split overlap for {case_id}: {membership[case_id]} and {name}")
            membership.setdefault(case_id, name)
    manifest = {
        "schema_version": 1,
        "updated_at": utc_now(),
        "status": "created_sealed",
        "sealed_final_test": True,
        "split_algorithm_sha256": hashlib.sha256(
            canonical_json({"version": 2, "algorithm": "template-family/entity/ood separated"})
        ).hexdigest(),
        "splits": records,
    }
    atomic_write_json(paths.state / "split_manifest.json", manifest)

    final_ids = set(splits["locked_final_test"])
    development = frame.loc[~frame["case_id"].isin(final_ids)].copy()
    development_target = paths.root / "data" / "silver" / "synthetic_cases_development.parquet"
    development_target.parent.mkdir(parents=True, exist_ok=True)
    temporary = development_target.with_suffix(".parquet.tmp")
    development.to_parquet(temporary, index=False)
    temporary.replace(development_target)
    manifest["development_projection"] = {
        "count": len(development),
        "sha256": sha256_file(development_target),
        "path": development_target.relative_to(paths.root).as_posix(),
    }
    atomic_write_json(paths.state / "split_manifest.json", manifest)


def load_split(name: str, *, phase: str) -> list[str]:
    paths = ProjectPaths.discover()
    if name == "locked_final_test" and not (phase == "P26" and os.environ.get("CONTROLFLOW_UNLOCK_FINAL") == "P26"):
        raise SealedTestAccessError("locked final test is accessible only by the P26 entry point")
    directory = "sealed" if name == "locked_final_test" else "splits"
    return (paths.root / "data" / directory / f"{name}.ids").read_text(encoding="utf-8").splitlines()


def run() -> str:
    paths = ProjectPaths.discover()
    master = paths.root / "data" / "sealed" / "benchmark_master.parquet"
    if not master.exists():
        raise FileNotFoundError("P08 sealed benchmark master is missing")
    with PhaseRun("P09", paths) as phase:
        frame = pd.read_parquet(master)
        if frame["case_id"].duplicated().any():
            raise ValueError("duplicate case IDs")
        normalized = (
            frame["narrative"]
            .str.casefold()
            .str.replace(r"reference\s+[a-z]+", "", regex=True)
            .str.replace(r"\d+(?:\.\d+)?", "#", regex=True)
            .str.replace(r"\s+", " ", regex=True)
        )
        duplicate_groups = normalized.groupby(normalized).groups
        proposed = create_splits(frame)
        membership = {
            case_id: name
            for name, identifiers in proposed.items()
            if name != "stratified_sanity"
            for case_id in identifiers
        }
        if any(
            len({membership.get(str(frame.loc[index, "case_id"])) for index in indices}) > 1
            for indices in duplicate_groups.values()
        ):
            raise ValueError("cross-split near-duplicate narratives detected")
        auto = frame[frame["expected_disposition"] == "AUTO"]
        if not auto["expected_actions"].map(lambda actions: set(actions).issubset(set(["propose_case_update"]))).all():
            raise ValueError("benchmark action truth is inconsistent")
        write_splits(frame)
        phase.register(paths.state / "split_manifest.json", "split_manifest")
        phase.register(
            paths.root / "data" / "silver" / "synthetic_cases_development.parquet",
            "development_dataset",
        )
    return str(paths.state / "split_manifest.json")


if __name__ == "__main__":
    print(run())
