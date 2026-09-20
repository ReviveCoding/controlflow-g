"""Fail-closed generation-to-run contamination binding verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from controlflow.core.state import atomic_write_json, sha256_file
from controlflow.v26.contamination_evidence import (
    config_digest,
    prior_inventory,
    read_evidence,
    semantic_digest,
)


def verify_transition(
    root: Path,
    *,
    candidate_path: Path,
    config_path: Path,
    inventory_path: Path,
    stored_path: Path,
    fresh_path: Path,
    manifest_path: Path,
    receipt_path: Path,
    persist: bool = True,
) -> dict[str, Any]:
    """Verify separately bound artifact integrity and semantic equivalence."""
    failures: list[str] = []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored_hash = sha256_file(stored_path)
    fresh_hash = sha256_file(fresh_path)
    stored = read_evidence(stored_path)
    fresh = read_evidence(fresh_path)
    runtime_hash = sha256_file(candidate_path)
    _, inventory, inventory_digest = prior_inventory(root, candidate_path, config_path)
    frozen_inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    checks = {
        "stored_report_sha256": stored_hash == manifest.get("contamination_sha256"),
        "stored_semantic_manifest_digest": stored.semantic_digest == manifest.get("contamination_semantic_digest"),
        "stored_semantic_self_check": stored.semantic_digest == semantic_digest(stored.semantic),
        "fresh_semantic_self_check": fresh.semantic_digest == semantic_digest(fresh.semantic),
        "fresh_semantic_equivalence": fresh.semantic_digest == stored.semantic_digest,
        "candidate_runtime_binding": runtime_hash
        == stored.semantic.candidate_runtime_sha256
        == fresh.semantic.candidate_runtime_sha256,
        "prior_inventory_binding": inventory_digest
        == stored.semantic.prior_inventory_digest
        == fresh.semantic.prior_inventory_digest,
        "prior_inventory_file_binding": sha256_file(inventory_path) == manifest.get("prior_inventory_sha256"),
        "prior_inventory_content": frozen_inventory == [item.model_dump() for item in inventory],
        "config_binding": config_digest(root, config_path)
        == stored.semantic.contamination_config_digest
        == fresh.semantic.contamination_config_digest,
        "distinct_recomputation": stored.provenance.run_id != fresh.provenance.run_id
        and stored.provenance.pid != fresh.provenance.pid,
        "zero_leakage": stored.semantic.leakage_findings == fresh.semantic.leakage_findings == 0,
    }
    failures.extend(key for key, passed in checks.items() if not passed)
    receipt = {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "stored_report_sha256": stored_hash,
        "fresh_recomputation_report_sha256": fresh_hash,
        "stored_semantic_digest": stored.semantic_digest,
        "fresh_semantic_digest": fresh.semantic_digest,
        "candidate_runtime_sha256": runtime_hash,
        "prior_inventory_digest": inventory_digest,
        "contamination_config_digest": config_digest(root, config_path),
        "stored_schema_version": stored.schema_version,
        "fresh_schema_version": fresh.schema_version,
        "semantic_schema_version": stored.semantic.schema_version,
    }
    if persist:
        atomic_write_json(receipt_path, receipt)
    return receipt
