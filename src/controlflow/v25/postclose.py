"""Independent post-close verification, executed as its own process."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sqlite3
import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml

from controlflow.core.state import atomic_write_json, canonical_json, sha256_file
from controlflow.v24.ledger_snapshot import TABLES
from controlflow.v25.artifact_closure import unbound_files, verify_bindings
from controlflow.v25.checkpoint_contract import CheckpointEnvelope
from controlflow.v25.checkpoint_integrity import expectations_from_evidence, verify_checkpoint
from controlflow.v25.denominators import recompute
from controlflow.v25.freeze_validation import verify_freeze


def _ledger_read_only(database: Path) -> dict[str, Any]:
    if not database.is_file():
        return {"valid": False, "failures": ["missing_database"]}
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in connection.execute("SELECT * FROM action_ledger ORDER BY sequence_id")]
        head = connection.execute("SELECT event_hash,sequence_id FROM ledger_head WHERE singleton=1").fetchone()
    finally:
        connection.close()
    failures: list[str] = []
    previous = "GENESIS"
    for sequence, row in enumerate(rows, start=1):
        if row["sequence_id"] != sequence or row["previous_event_hash"] != previous:
            failures.append(f"sequence_or_predecessor:{sequence}")
        payload = {key: value for key, value in row.items() if key not in {"sequence_id", "event_hash"}}
        for key in ("review_required", "approval_valid", "committed"):
            payload[key] = bool(payload[key])
        if hashlib.sha256(canonical_json(payload)).hexdigest() != row["event_hash"]:
            failures.append(f"event_hash:{sequence}")
        previous = row["event_hash"]
    if rows and (head is None or head["event_hash"] != previous or head["sequence_id"] != len(rows)):
        failures.append("head")
    if not rows and head is not None:
        failures.append("empty_ledger_head")
    return {"valid": not failures, "failures": failures, "event_count": len(rows), "head": dict(head) if head else None}


def _snapshot_read_only(database: Path) -> dict[str, list[dict[str, Any]]]:
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        return {
            name: [dict(row) for row in connection.execute(f"SELECT * FROM {name} ORDER BY rowid")] for name in TABLES
        }
    finally:
        connection.close()


def verify(root: Path, graph_path: Path, output_path: Path) -> dict[str, Any]:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    role = graph["role"]
    failures = verify_bindings(root, graph)
    failures.extend(f"unbound:{item}" for item in unbound_files(root, graph))
    bindings = graph["bindings"]
    required = {
        "runtime_cases",
        "evaluator_truth",
        "evidence_corpus",
        "authorization_state",
        "dataset_manifest",
        "stratum_manifest",
        "denominator_contract",
        "structural_admission",
        "contamination_report",
        "candidate_output",
        "per_case_metrics",
        "aggregate_metrics",
        "denominator_report",
        "server_preflight",
        "checkpoint",
        "durable_candidate_stream",
        "candidate_manifest",
        "environment_manifest",
        "gate_config",
        "freeze_manifest",
        "qualification_sqlite",
        "sqlite_finalization",
        "sqlite_stability",
        "ledger_verification",
        "ledger_snapshot",
    }
    failures.extend(f"missing_logical_binding:{name}" for name in sorted(required - set(bindings)))
    if failures:
        report = {"schema_version": 1, "status": "FAIL", "failures": failures}
        atomic_write_json(output_path, report)
        return report

    def payload(name: str) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads((root / bindings[name]["path"]).read_text(encoding="utf-8")))

    finalized, stability = payload("sqlite_finalization"), payload("sqlite_stability")
    database = root / bindings["qualification_sqlite"]["path"]
    checkpoint = finalized.get("checkpoint")
    if (
        finalized.get("status") != "FINALIZED"
        or finalized.get("exit_code") != 0
        or finalized.get("journal_mode") != "wal"
        or finalized.get("integrity_check") != "ok"
        or finalized.get("process_identity") != "controlflow.v24.sqlite_finalization"
        or not isinstance(finalized.get("pid"), int)
        or not isinstance(finalized.get("sqlite_version"), str)
        or not isinstance(checkpoint, list)
        or len(checkpoint) != 3
        or checkpoint[0] != 0
        or checkpoint[1] != checkpoint[2]
        or finalized.get("wal_bytes_after_close") != 0
        or finalized.get("shm_bytes_after_close") != 0
    ):
        failures.append("finalizer_invalid")
    strata_name = "rehearsal_strata.yaml" if role == "development" else "strata.yaml"
    interval = yaml.safe_load((root / "configs/v25" / strata_name).read_text(encoding="utf-8"))[
        "stability_interval_seconds"
    ]
    if (
        stability.get("status") != "STABLE"
        or stability.get("exit_code") != 0
        or stability.get("process_identity") != "controlflow.v24.hash_stability"
        or stability.get("interval_seconds") != interval
        or stability.get("sha256_1") != stability.get("sha256_2")
        or stability.get("size_1") != stability.get("size_2")
    ):
        failures.append("stability_invalid")
    if (
        stability.get("sha256_2") != bindings["qualification_sqlite"]["sha256"]
        or stability.get("size_2") != bindings["qualification_sqlite"]["size"]
    ):
        failures.append("sqlite_binding_stability_mismatch")
    ledger = _ledger_read_only(database)
    if not ledger["valid"] or ledger != payload("ledger_verification"):
        failures.append("ledger_verification_invalid")
    if finalized.get("ledger_event_count") != ledger["event_count"] or finalized.get("ledger_head") != ledger["head"]:
        failures.append("finalizer_ledger_mismatch")
    for suffix in ("-wal", "-shm"):
        sidecar = database.with_name(database.name + suffix)
        if sidecar.is_file() and sidecar.stat().st_size:
            failures.append(f"sqlite_sidecar_persisted:{suffix}")
    snapshot = payload("ledger_snapshot").get("tables", {})
    if snapshot != _snapshot_read_only(database):
        failures.append("ledger_snapshot_database_mismatch")
    if sha256_file(database) != bindings["qualification_sqlite"]["sha256"]:
        failures.append("sqlite_mutated_during_verification")
    dataset = payload("dataset_manifest")
    for name, key in (
        ("runtime_cases", "runtime"),
        ("evaluator_truth", "truth"),
        ("evidence_corpus", "evidence"),
        ("authorization_state", "authorization"),
        ("stratum_manifest", "stratum_manifest"),
    ):
        if dataset.get(f"{key}_sha256") != bindings[name]["sha256"]:
            failures.append(f"dataset_hash:{name}")
    stratum = payload("stratum_manifest")
    denominator = payload("denominator_report")
    if stratum.get("counts") != denominator.get("independent_stratum_counts") or denominator.get("status") != "VALID":
        failures.append("denominator_or_stratum_invalid")
    freeze: dict[str, Any] = {}
    try:
        freeze = verify_freeze(root, role=role)
        if sha256_file(root / bindings["gate_config"]["path"]) != freeze["gate_config_sha256"]:
            failures.append("gate_freeze_mismatch")
        candidate_manifest = payload("candidate_manifest")
        dataset_manifest = payload("dataset_manifest")
        checkpoint_path = root / bindings["checkpoint"]["path"]
        stream_path = root / bindings["durable_candidate_stream"]["path"]
        expected = expectations_from_evidence(
            candidate_threshold=float(candidate_manifest["critical_threshold"]),
            candidate_bundle_hash=str(freeze["freeze_hash"]),
            gate_config_path=root / bindings["gate_config"]["path"],
            freeze_path=root / bindings["freeze_manifest"]["path"],
            runtime_path=root / bindings["runtime_cases"]["path"],
            evidence_path=root / bindings["evidence_corpus"]["path"],
            authorization_path=root / bindings["authorization_state"]["path"],
            qwen_revision=str(freeze["qwen_revision"]),
            vllm_version=str(freeze["vllm_version"]),
            seed=int(dataset_manifest["seed"]),
            sample_count=int(dataset_manifest["count"]),
        )
        checkpoint_result = verify_checkpoint(checkpoint_path, stream_path, expected)
        failures.extend(checkpoint_result["failures"])
        checkpoint_view = CheckpointEnvelope.load(checkpoint_path)
        recalculated = recompute(
            root,
            (root / bindings["runtime_cases"]["path"]).parent,
            root / bindings["candidate_output"]["path"],
            root / bindings["per_case_metrics"]["path"],
            root / bindings["aggregate_metrics"]["path"],
            root / bindings["ledger_snapshot"]["path"],
            None,
            critical_threshold=checkpoint_view.critical_threshold,
            role=role.upper(),
        )
        if recalculated != denominator:
            failures.append("denominator_recomputation_mismatch")
    except (KeyError, RuntimeError, ValueError, OSError) as exc:
        failures.append(f"freeze_or_denominator_verification:{exc}")
    if (
        payload("structural_admission").get("status") != "ADMITTED"
        or payload("contamination_report").get("leakage_findings") != 0
    ):
        failures.append("admission_or_contamination_invalid")
    server = payload("server_preflight")
    provenance = server.get("server_provenance", {})
    frozen_serving = freeze.get("serving", {})
    expected_argv = [
        "/home/bjw-0/.venvs/controlflow-g-v2/bin/python",
        "/home/bjw-0/.venvs/controlflow-g-v2/bin/vllm",
        "serve",
        freeze.get("qwen_model"),
        "--revision",
        freeze.get("qwen_revision"),
        "--served-model-name",
        "controlflow-g-v23-qwen3-4b",
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        "0.72",
        "--max-model-len",
        str(frozen_serving.get("max_model_len")),
        "--max-num-seqs",
        str(frozen_serving.get("max_num_seqs")),
        "--structured-outputs-config.backend",
        "xgrammar",
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--enable-per-request-metrics",
        "--enable-request-id-headers",
        "--performance-mode",
        "interactivity",
        "--optimization-level",
        "2",
        "--max-num-batched-tokens",
        "2048",
        "--generation-config",
        "vllm",
        "--host",
        str(frozen_serving.get("host")),
        "--port",
        str(frozen_serving.get("port")),
    ]
    command = str(provenance.get("process_command", ""))
    actual_argv = shlex.split(command.split(maxsplit=1)[1]) if " " in command else []
    cold = server.get("cold_metrics", {})
    cold_sums = {
        name: sum(float(item["value"]) for item in cold.get("samples", []) if item.get("name") == name)
        for name in ("vllm:request_success_total", "vllm:prefix_cache_queries_total")
    }
    gpu = server.get("gpu_ownership", {})
    if (
        not provenance.get("verified")
        or provenance.get("run_role") != role
        or provenance.get("vllm_version") != freeze.get("vllm_version")
        or str(freeze.get("qwen_model")) not in command
        or actual_argv != expected_argv
        or f"V23_RUN_ROLE={role}" not in provenance.get("environment", [])
        or "VLLM_USE_V2_MODEL_RUNNER=0" not in provenance.get("environment", [])
        or any(value != 0 for value in cold_sums.values())
        or gpu.get("server_pid") != provenance.get("pid")
        or not gpu.get("all_server_descendants")
        or not gpu.get("gpu_compute_pids")
    ):
        failures.append("server_preflight_invalid")
    if not payload("environment_manifest"):
        failures.append("checkpoint_or_environment_invalid")
    if not pd.read_parquet(root / bindings["candidate_output"]["path"]).case_id.is_unique:
        failures.append("candidate_output_duplicates")
    report = {"schema_version": 1, "status": "PASS" if not failures else "FAIL", "failures": failures, "ledger": ledger}
    atomic_write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("graph", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = verify(args.root, args.graph, args.report)
    if report["status"] != "PASS":
        print(f"POSTCLOSE_VERIFICATION_FAILED:{report['failures']}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
