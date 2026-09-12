from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import joblib
import yaml
from filelock import FileLock

from controlflow.core.state import PhaseRun, ProjectPaths, atomic_write_json, canonical_json, sha256_file, utc_now


class FreezeViolation(RuntimeError):
    pass


def freeze_identity_hash(manifest: dict[str, Any]) -> str:
    identity = {key: value for key, value in manifest.items() if key not in {"freeze_hash", "created_at"}}
    return hashlib.sha256(canonical_json(identity)).hexdigest()


def _git_revision(paths: ProjectPaths) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=paths.root, check=True, capture_output=True, text=True, timeout=10
    )
    if subprocess.run(
        ["git", "status", "--porcelain"], cwd=paths.root, check=True, capture_output=True, text=True, timeout=10
    ).stdout.strip():
        raise FreezeViolation("candidate tree must be clean before freeze")
    return result.stdout.strip()


def create_freeze() -> str:
    paths = ProjectPaths.discover()
    state = json.loads(paths.execution_state.read_text(encoding="utf-8"))
    if "P24" not in state["completed_phases"]:
        raise FreezeViolation("P24 stable-snapshot review must complete before freeze")
    if state["sealed_test_consumed"]:
        raise FreezeViolation("sealed test has already been consumed")
    artifact = paths.root / "artifacts/frozen_risk_service.joblib"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    calibrated = paths.root / "artifacts/calibrated_risk_service.joblib"
    if not calibrated.exists():
        raise FreezeViolation("P12 calibrated risk artifact is missing")
    joblib.dump(joblib.load(calibrated), artifact)
    required = [
        Path("uv.lock"),
        Path("pyproject.toml"),
        Path("state/data_manifest.json"),
        Path("state/split_manifest.json"),
        Path("data/silver/synthetic_cases_development.parquet"),
        Path("data/sealed/benchmark_master.parquet"),
        Path("data/sealed/locked_final_test.ids"),
        Path("data/sealed/locked_final_test.parquet"),
        Path("data/sealed/locked_final_test.metadata.json"),
        Path("data/staging/nist_controls_raw.parquet"),
        Path("data/staging/cfr_raw.parquet"),
        Path("data/staging/transactions_raw.parquet"),
        Path("results/agents.parquet"),
        Path("results/security.parquet"),
        Path("artifacts/frozen_risk_service.joblib"),
    ]
    required.extend(path.relative_to(paths.root) for path in sorted((paths.root / "configs").rglob("*.yaml")))
    required.extend(path.relative_to(paths.root) for path in sorted((paths.root / "src/controlflow").rglob("*.py")))
    entries = [
        {"path": item.as_posix(), "sha256": sha256_file(paths.root / item), "bytes": (paths.root / item).stat().st_size}
        for item in required
    ]
    identity: dict[str, Any] = {
        "schema_version": 1,
        "git_revision": _git_revision(paths),
        "candidate": "AG6_controlflow_g",
        "llm_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "llm_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "embedding_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "agent_graph_version": "agent-graph-v3",
        "authorization_policy_version": "local-policy-v1",
        "tool_schema_version": 1,
        "seeds": [17],
        "artifacts": entries,
    }
    payload = {**identity, "created_at": utc_now()}
    payload["freeze_hash"] = freeze_identity_hash(payload)
    target = paths.state / "freeze_manifest.json"
    atomic_write_json(target, payload)
    with PhaseRun("P25", paths) as phase:
        phase.register(artifact, "frozen_model")
        phase.register(target, "freeze_manifest")
    with FileLock(str(paths.execution_state) + ".lock"):
        state = json.loads(paths.execution_state.read_text(encoding="utf-8"))
        state["freeze_hash"] = payload["freeze_hash"]
        state["updated_at"] = utc_now()
        atomic_write_json(paths.execution_state, state)
    return str(target)


def verify_frozen_artifacts(root: Path, artifacts: list[dict[str, Any]]) -> None:
    for item in artifacts:
        target = root / item["path"]
        if not target.is_file() or sha256_file(target) != item["sha256"]:
            raise FreezeViolation(f"frozen artifact changed: {item['path']}")


def verify_freeze(paths: ProjectPaths) -> dict[str, Any]:
    manifest = json.loads((paths.state / "freeze_manifest.json").read_text(encoding="utf-8"))
    claimed = manifest["freeze_hash"]
    actual = freeze_identity_hash(manifest)
    if claimed != actual:
        raise FreezeViolation("freeze manifest hash mismatch")
    verify_frozen_artifacts(paths.root, manifest["artifacts"])
    if _git_revision(paths) != manifest["git_revision"]:
        raise FreezeViolation("Git revision differs from frozen candidate")
    return cast(dict[str, Any], manifest)


def consume_seal(paths: ProjectPaths) -> None:
    """Legacy non-resumable seal transition; retained for explicit callers."""
    lock = FileLock(str(paths.execution_state) + ".lock")
    with lock:
        state = json.loads(paths.execution_state.read_text(encoding="utf-8"))
        if state["sealed_test_consumed"]:
            raise FreezeViolation("locked final test may be evaluated only once")
        state["sealed_test_consumed"] = True
        state["sealed_test_consumed_at"] = utc_now()
        state["updated_at"] = utc_now()
        atomic_write_json(paths.execution_state, state)


def begin_or_resume_final_run(paths: ProjectPaths, freeze_hash: str) -> dict[str, Any]:
    """Atomically authorize one freeze-bound P26 run, including crash resumes.

    The durable run descriptor is written before the seal is consumed. Once
    consumed, only that same run and exact freeze may resume.
    """
    run_path = paths.state / "final_run.json"
    lock = FileLock(str(paths.execution_state) + ".lock")
    with lock:
        state = json.loads(paths.execution_state.read_text(encoding="utf-8"))
        existing = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else None
        if existing is not None and (
            existing.get("freeze_hash") != freeze_hash
            or existing.get("run_id") != hashlib.sha256(f"P26:{freeze_hash}".encode()).hexdigest()
        ):
            raise FreezeViolation("final-run descriptor does not match the frozen candidate")
        if state["sealed_test_consumed"]:
            if existing is None or existing.get("status") not in {"in_progress", "complete"}:
                raise FreezeViolation("consumed holdout has no resumable freeze-bound run")
            if existing["status"] == "in_progress":
                prior_started = datetime.fromisoformat(str(existing["attempt_started_at"]))
                accumulated = float(existing.get("accumulated_runtime_seconds", 0.0))
                accumulated += max(0.0, (datetime.now(UTC) - prior_started).total_seconds())
                existing.update(
                    accumulated_runtime_seconds=accumulated,
                    retry_count=int(existing.get("retry_count", 0)) + 1,
                    attempt_started_at=utc_now(),
                    updated_at=utc_now(),
                )
                atomic_write_json(run_path, existing)
            return cast(dict[str, Any], existing)
        if existing is None:
            existing = {
                "run_id": hashlib.sha256(f"P26:{freeze_hash}".encode()).hexdigest(),
                "freeze_hash": freeze_hash,
                "status": "in_progress",
                "created_at": utc_now(),
                "attempt_started_at": utc_now(),
                "accumulated_runtime_seconds": 0.0,
                "retry_count": 0,
                "updated_at": utc_now(),
            }
            atomic_write_json(run_path, existing)
        elif existing.get("status") != "in_progress":
            raise FreezeViolation("a completed final run cannot consume a new holdout")
        state["sealed_test_consumed"] = True
        state["sealed_test_consumed_at"] = utc_now()
        state["final_run_id"] = existing["run_id"]
        state["updated_at"] = utc_now()
        atomic_write_json(paths.execution_state, state)
        return cast(dict[str, Any], existing)


def complete_final_run(paths: ProjectPaths, run_id: str, result_hash: str) -> None:
    run_path = paths.state / "final_run.json"
    with FileLock(str(paths.execution_state) + ".lock"):
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run.get("run_id") != run_id or run.get("status") != "in_progress":
            raise FreezeViolation("final-run completion does not match the active run")
        started = datetime.fromisoformat(str(run["attempt_started_at"]))
        accumulated = float(run.get("accumulated_runtime_seconds", 0.0))
        accumulated += max(0.0, (datetime.now(UTC) - started).total_seconds())
        run.update(
            status="complete",
            result_sha256=result_hash,
            accumulated_runtime_seconds=accumulated,
            updated_at=utc_now(),
        )
        atomic_write_json(run_path, run)


def evaluate_release_gates(metrics: dict[str, float]) -> tuple[str, list[dict[str, Any]]]:
    config = yaml.safe_load((ProjectPaths.discover().root / "configs/release_gates.yaml").read_text(encoding="utf-8"))
    decisions = []
    for name, gate in config["gates"].items():
        value = metrics.get(name)
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            passed = False
        elif gate["operator"] == "eq":
            passed = value == gate["threshold"]
        elif gate["operator"] == "gte":
            passed = value >= gate["threshold"]
        elif gate["operator"] == "lte":
            passed = value <= gate["threshold"]
        else:
            raise ValueError(f"unsupported gate operator: {gate['operator']}")
        decisions.append({"gate": name, "value": value, "passed": passed, **gate})
    zero_failure = any(not row["passed"] and row.get("tolerance") == "zero" for row in decisions)
    failures = sum(not row["passed"] for row in decisions)
    invalid = any(
        row["value"] is None or not isinstance(row["value"], (int, float)) or not math.isfinite(float(row["value"]))
        for row in decisions
    )
    decision = (
        "NO_PROMOTE"
        if invalid or zero_failure or failures >= 2
        else "CONDITIONAL_PROMOTE"
        if failures == 1
        else "PROMOTE"
    )
    return decision, decisions


if __name__ == "__main__":
    print(create_freeze())
