"""Admission checks that run before candidate access to a holdout."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from controlflow.core.state import atomic_write_json, sha256_file
from controlflow.v22.dgp import RUNTIME_FORBIDDEN
from controlflow.v24.opportunities import certify_opportunities
from controlflow.v25.generator import STALE_ALTERNATIVES


def _time(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def temporal_oracle(event_time: object, system_time: object, policies: list[dict[str, Any]]) -> str | None:
    event = _time(event_time)
    system = _time(system_time)
    eligible = [
        item
        for item in policies
        if _time(item["business_valid_from"]) <= event
        and (item["business_valid_to"] is None or event < _time(item["business_valid_to"]))
        and _time(item["system_known_from"]) <= system
    ]
    if not eligible:
        return None
    maximum = max(int(item["correction_rank"]) for item in eligible)
    best = [item for item in eligible if int(item["correction_rank"]) == maximum]
    return str(best[0]["policy_id"]) if len(best) == 1 else None


def independent_counts(
    runtime: pd.DataFrame, truth: pd.DataFrame, policies: list[dict[str, Any]]
) -> tuple[dict[str, int], list[str]]:
    failures: list[str] = []
    joined = truth.merge(runtime, on="case_id", validate="one_to_one", suffixes=("_truth", "_runtime"))
    policy_ids = {str(row["policy_id"]) for row in policies}
    by_id = {str(row["policy_id"]): row for row in policies}
    stale_count = 0
    for row in joined.itertuples(index=False):
        event_time = row.event_time_runtime
        system_time = row.system_time_runtime
        oracle = temporal_oracle(event_time, system_time, policies)
        expected = row.expected_policy_id
        if pd.isna(expected):
            expected = None
        if oracle != expected:
            failures.append(f"temporal_oracle:{row.case_id}")
        if row.event_time_truth != event_time or row.system_time_truth != system_time:
            failures.append(f"truth_time_mismatch:{row.case_id}")
        opportunity = row.temporal_scenario in STALE_ALTERNATIVES and expected is not None
        if bool(row.is_stale_challenge) != opportunity:
            failures.append(f"stale_opportunity_marker_mismatch:{row.case_id}")
        if not opportunity and len(row.stale_candidate_ids):
            failures.append(f"unexpected_stale_alternatives:{row.case_id}")
        if opportunity:
            alternatives = set(row.stale_candidate_ids)
            declared = set(STALE_ALTERNATIVES.get(row.temporal_scenario, ()))
            if not expected or not alternatives or alternatives != declared or expected in alternatives:
                failures.append(f"invalid_stale_challenge:{row.case_id}")
            elif expected not in policy_ids or not alternatives.issubset(policy_ids):
                failures.append(f"missing_candidate_accessible_policy:{row.case_id}")
            else:
                correct = by_id[str(expected)]
                meaningful = any(
                    (
                        by_id[identifier]["business_valid_from"] != correct["business_valid_from"]
                        or by_id[identifier]["business_valid_to"] != correct["business_valid_to"]
                        or by_id[identifier]["system_known_from"] != correct["system_known_from"]
                        or by_id[identifier]["correction_rank"] != correct["correction_rank"]
                    )
                    and identifier != oracle
                    for identifier in alternatives
                )
                if not meaningful:
                    failures.append(f"noncompeting_policy_alternatives:{row.case_id}")
                else:
                    stale_count += 1
    latent = truth.latent.map(json.loads)
    counts = {
        "critical": int(truth.truth_critical.astype(bool).sum()),
        "require_review": int((truth.truth_disposition == "REQUIRE_REVIEW").sum()),
        "deny": int((truth.truth_disposition == "DENY").sum()),
        "stale_policy": stale_count,
        "historical_temporal": int((runtime.event_time.astype(str).str[:4] < "2026").sum()),
        "missing_evidence": int(latent.map(lambda item: item["evidence_quality"] < 0.30).sum()),
        "conflicting_evidence": int(latent.map(lambda item: item["conflict_state"]).sum()),
        "scope_restriction": int((runtime.requested_scope == "restricted").sum()),
        "adversarial": int(truth.is_adversarial.astype(bool).sum()),
        "routine": int((truth.truth_disposition == "ALLOW").sum()),
    }
    return counts, failures


def structural_admission(
    directory: Path,
    root: Path,
    contamination: dict[str, Any],
    output: Path | None = None,
    *,
    role: str = "QUALIFICATION",
) -> dict[str, Any]:
    if role not in {"DEVELOPMENT", "QUALIFICATION", "FINAL"}:
        raise ValueError("unknown admission role")
    expected_count = 60 if role == "DEVELOPMENT" else 600
    failures: list[str] = []
    required = {
        "runtime": directory / "runtime_cases.parquet",
        "truth": directory / "evaluator_truth.parquet",
        "evidence": directory / "evidence_corpus.parquet",
        "authorization": directory / "authorization_state.parquet",
        "stratum_manifest": directory / "stratum_manifest.json",
    }
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file() or any(not path.is_file() for path in required.values()):
        result = {"schema_version": 1, "status": "V25_GENERATION_INVALID", "failures": ["missing_required_file"]}
        if output is not None:
            atomic_write_json(output, result)
        return result
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    strata_file = "rehearsal_strata.yaml" if role == "DEVELOPMENT" else "strata.yaml"
    strata_config = yaml.safe_load((root / "configs/v25" / strata_file).read_text(encoding="utf-8"))
    seed_key = "development_seed" if role == "DEVELOPMENT" else f"{role.lower()}_seed"
    expected_seed = int(strata_config[seed_key])
    expected_prefix = {
        "DEVELOPMENT": f"V25DEV{expected_seed}",
        "QUALIFICATION": "V25QUAL",
        "FINAL": "V25FINAL",
    }[role]
    if (
        manifest.get("role") != role
        or type(manifest.get("seed")) is not int
        or manifest["seed"] != expected_seed
        or manifest.get("prefix") != expected_prefix
        or manifest.get("generator") != "controlflow.v25.generator.generate_v25_split"
        or type(manifest.get("count")) is not int
        or manifest["count"] != expected_count
    ):
        failures.append("dataset_identity_or_seed")
    for name, path in required.items():
        if manifest.get(f"{name}_sha256") != sha256_file(path):
            failures.append(f"hash_mismatch:{name}")
    runtime = pd.read_parquet(required["runtime"])
    truth = pd.read_parquet(required["truth"])
    authorization = pd.read_parquet(required["authorization"])
    evidence = pd.read_parquet(required["evidence"])
    opportunities = certify_opportunities(root, runtime, evidence, authorization)
    if (
        len(runtime) != expected_count
        or len(truth) != expected_count
        or runtime.case_id.duplicated().any()
        or truth.case_id.duplicated().any()
    ):
        failures.append("case_cardinality")
    if set(runtime.case_id) != set(truth.case_id):
        failures.append("case_id_mismatch")
    if any(not str(case_id).startswith(expected_prefix + "-") for case_id in runtime.case_id):
        failures.append("case_id_prefix")
    forbidden = RUNTIME_FORBIDDEN | {
        "is_stale_challenge",
        "stale_candidate_ids",
        "temporal_scenario",
        "expected_policy_id",
        "truth_critical",
    }
    if set(runtime.columns) & forbidden:
        failures.append("runtime_truth_leakage")
    if required["runtime"].resolve() == required["truth"].resolve():
        failures.append("runtime_truth_not_separate")
    if len(authorization) != expected_count or len(set(authorization.identity.astype(str))) != expected_count:
        failures.append("authorization_cardinality")
    if evidence.document_id.duplicated().any():
        failures.append("evidence_duplicate_document_id")
    policies = yaml.safe_load((root / "configs/v22/temporal_policies.yaml").read_text(encoding="utf-8"))["policies"]
    counts, oracle_failures = independent_counts(runtime, truth, policies)
    failures.extend(oracle_failures)
    stratum = json.loads(required["stratum_manifest"].read_text(encoding="utf-8"))
    if stratum.get("counts") != counts:
        failures.append("stratum_count_drift")
    config = strata_config
    for name, observed in counts.items():
        required_count = int(config[f"{name}_minimum"])
        if observed < required_count:
            failures.append(f"minimum:{name}:{observed}<{required_count}")
    contract_name = "rehearsal_denominator_contract.yaml" if role == "DEVELOPMENT" else "denominator_contract.yaml"
    contract = yaml.safe_load((root / "configs/v25" / contract_name).read_text(encoding="utf-8"))
    expected_evidence = int(truth.expected_evidence_ids.map(len).sum())
    if expected_evidence < int(contract["metrics"]["evidence_completeness"]["minimum_denominator"]):
        failures.append("minimum:expected_evidence")
    if contamination.get("leakage_findings") != 0:
        failures.append("contamination")
    if opportunities["review_certified_count"] < int(
        contract["metrics"]["approval_bypass_rate"]["minimum_denominator"]
    ):
        failures.append("minimum:authoritative_require_review_certificate")
    if opportunities["deny_certified_count"] < int(
        contract["metrics"]["unauthorized_commit_rate"]["minimum_denominator"]
    ):
        failures.append("minimum:authoritative_deny_certificate")
    result = {
        "schema_version": 1,
        "status": "ADMITTED" if not failures else "V25_GENERATION_INVALID",
        "count": len(runtime),
        "independent_counts": counts,
        "expected_evidence_count": expected_evidence,
        "authoritative_opportunity_certificate": opportunities,
        "failures": failures,
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "stratum_manifest_sha256": sha256_file(required["stratum_manifest"]),
    }
    if output is not None:
        atomic_write_json(output, result)
    return result
