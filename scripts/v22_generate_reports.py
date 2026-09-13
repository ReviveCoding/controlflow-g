from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from controlflow.core.state import atomic_write_json, sha256_file, utc_now

ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict[str, Any] | None:
    target = ROOT / path
    return json.loads(target.read_text(encoding="utf-8")) if target.exists() else None


def _metric(metrics: dict[str, Any] | None, name: str) -> str:
    if not metrics or name not in metrics:
        return "not measured"
    value = metrics[name]
    if isinstance(value, dict) and "numerator" in value:
        estimate = value["estimate"]
        rendered = "undefined" if estimate is None else f"{estimate:.4f}"
        return f"{value['numerator']}/{value['denominator']} ({rendered})"
    return str(value)


def _write(name: str, title: str, body: str) -> None:
    path = ROOT / f"reports/v22/{name}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.strip()}\n", encoding="utf-8")


def main() -> None:
    execution = _load("state/v22_execution_state.json") or {}
    dataset = _load("state/v22_dataset_manifest.json") or {}
    bundle = _load("state/v22_model_bundle.json") or {}
    reviews = _load("state/v22_review_findings.json") or {}
    qualification = _load("state/v22_qualification_manifest.json") or {}
    freeze = _load("state/v22_freeze_manifest.json") or {}
    metrics_binding = execution.get("development_metrics", {})
    metrics = _load(metrics_binding.get("path", "")) if metrics_binding else None
    model_path = Path(bundle.get("critical_model", {}).get("path", "missing")).parent / "training_report.json"
    training = _load(model_path.as_posix())
    serving = _load("state/v22_vllm_structured_c2.json")
    integrity = _load("state/v22_integrity.json")
    ablation = _load("results/v22/development_ablations.json")
    _write(
        "01_data_independence.md",
        "V2.2 Data Independence",
        f"Only {', '.join(dataset.get('roles_present', [])) or 'no'} development roles exist. Qualification present: "
        f"{dataset.get('qualification_present', False)}. Runtime files exclude the forbidden truth fields and the "
        "evaluator truth is physically separate. Split seeds, template families, paths, and SHA256 values are in "
        "`state/v22_dataset_manifest.json`; automated exact and character-ngram similarity results are bound there.",
    )
    _write(
        "02_model_training.md",
        "V2.2 Model Training",
        "Models consume runtime-observable numeric, categorical, and narrative features. TRAIN fits the models; "
        "VALIDATION selects architectures. GPU claims are limited to the device evidence in the machine report. "
        "The bundle explicitly binds tabular inference to CPU so live WSL vLLM exclusively owns VRAM.\n\n"
        f"Selected critical model: {None if training is None else training.get('selected_critical_model')}. "
        f"Selected noncritical model: {None if training is None else training.get('selected_noncritical_model')}. "
        f"GPU evidence: {None if training is None else training.get('gpu_evidence')}.",
    )
    _write(
        "03_calibration_threshold.md",
        "V2.2 Calibration and Critical Threshold",
        "Calibration and threshold selection use CALIBRATION only. The predeclared rule retains thresholds with "
        "critical recall at least 0.95, then minimizes FPR plus 1.5 times residual critical risk and a penalty "
        "for falling below a 0.98 development safety margin; ties favor the higher threshold.\n\n"
        f"Selected calibrator: {None if training is None else training.get('selected_calibrator')}; threshold: "
        f"{bundle.get('critical_threshold')}; provenance: {bundle.get('threshold_provenance')}.",
    )
    _write(
        "04_candidate_bundle.md",
        "V2.2 Candidate Bundle",
        "Runtime loads models and configuration exclusively through `state/v22_model_bundle.json`. Each file binding "
        f"has a SHA256 and verification fails closed. Bundle hash: `{bundle.get('bundle_sha256', 'not created')}`.",
    )
    _write(
        "05_pdp_pep.md",
        "V2.2 PDP and PEP",
        "The PEP accepts a proposed action and authenticated context, canonicalizes the action, and obtains its own "
        "authoritative PDP decision. The explicit registry denies unknown actions. The PDP is queried again inside "
        "the commit transaction using current authorization state.",
    )
    _write(
        "06_signed_approvals.md",
        "V2.2 Signed Approvals",
        "The separate ApprovalIssuer holds an ignored local Ed25519 private key. The bundle and PEP bind only the "
        "public key. Tokens bind case, action, policy decision, authorization snapshot, versions, reviewer, expiry, "
        "and nonce; tampering and replay are negative-tested.",
    )
    _write(
        "07_transactional_executor.md",
        "V2.2 Transactional Executor",
        "SQLite stores case_state, policy_decisions, action_ledger, and approval_consumption. BEGIN IMMEDIATE covers "
        "revalidation, signature/binding verification, state mutation, hash-chain append, token consumption, and "
        "commit. The structure is a tamper-evident local audit chain, not a distributed immutable ledger.\n\n"
        f"Security metrics: {None if metrics is None else metrics.get('security')}; ledger audit: "
        f"{None if metrics is None else metrics.get('ledger_audit')}.",
    )
    _write(
        "08_temporal_oracle.md",
        "V2.2 Temporal Oracle",
        "Evaluator expected policy IDs are loaded from an evaluator-only declarative fixture file. Candidate "
        "temporal retrieval is a "
        "separate bitemporal corpus implementation with stale, future, correction, current-version, and irrelevant "
        f"distractors. Development temporal accuracy: {_metric(metrics, 'temporal_policy_accuracy')}.",
    )
    _write(
        "09_vllm_serving.md",
        "V2.2 vLLM Serving",
        "No proxy is substituted. "
        + (
            "Live serving has not yet produced admissible evidence."
            if serving is None
            else f"Live vLLM evidence records {serving['requests']} requests at concurrency={serving['concurrency']}, "
            f"structured failures={serving['total_structured_failures']}, and total P95="
            f"{serving['total_p95_seconds']:.3f}s."
        ),
    )
    _write(
        "10_checkpoint_integrity.md",
        "V2.2 Checkpoint and Integrity",
        "The actual dataset runner writes and validates a fingerprint before partial output. Mismatch raises "
        "CHECKPOINT_INCOMPATIBLE. Integrity is derived from contamination, tests, bundle verification, checkpoint "
        f"artifacts, ledger audit, reviews, and live serving. Current integrity: {integrity}.",
    )
    _write(
        "11_ablation.md",
        "V2.2 Paired Ablations",
        "Ablations rerun affected candidate, policy, retrieval, PEP, and executor components against paired cases. "
        f"Current artifact: {'not run' if ablation is None else 'results/v22/development_ablations.json'}.",
    )
    _write(
        "12_review_disposition.md",
        "V2.2 Review Disposition",
        f"V2.1 findings are ingested with repair/evidence/test fields. Review status: "
        f"{reviews.get('review_status', 'NOT_RUN')}; unresolved counts: {reviews.get('counts')}.",
    )
    _write(
        "13_internal_qualification.md",
        "V2.2 Internal Qualification",
        f"Status: {qualification.get('status', 'NOT_GENERATED')}. "
        f"Reason: {qualification.get('reason', 'qualification has not been authorized')}.",
    )
    _write(
        "14_final_evaluation.md",
        "V2.2 Final Evaluation",
        "No V2.2 final holdout has been generated or consumed. Final evaluation is prohibited until qualification "
        "passes, post-qualification reviewers clear, and a clean deterministic freeze exists.",
    )
    _write(
        "15_release_decision.md",
        "V2.2 Release Decision",
        "No frozen release decision exists. A decision will be emitted only after the required qualification, "
        f"post-review, freeze, and one-shot final protocol. Freeze status: {freeze.get('status', 'NOT_FROZEN')}.",
    )
    _write(
        "16_limitations.md",
        "V2.2 Limitations",
        "This is a local simulated study, not a bank deployment and not authorized for real financial action. A "
        "local SHA256 chain is tamper-evident only while its head anchor is trusted. Synthetic latent labels do not "
        "establish production validity. Observed zero security failures cannot prove zero true risk. Rationale "
        "diagnostics remain secondary to typed Core STC.",
    )
    _write(
        "17_future_work.md",
        "V2.2 Future Work",
        "Externalize audit-head trust and signing keys, broaden public-data grounding, add independent human rationale "
        "judgments, and evaluate drift in a production-like simulation. QLoRA remains out of scope unless the "
        "architecture and security gates clear and measured semantic quality is the remaining bottleneck.",
    )
    technical = ROOT / "TECHNICAL_REPORT_V22.md"
    technical.write_text(
        "# ControlFlow-G V2.2 Technical Report\n\n"
        "V2.2 separates latent truth generation, fitted candidate inference, evidence and temporal retrieval, "
        "authoritative policy decision, enforcement, external signed review, atomic simulated state, local "
        "tamper-evident audit, and evaluator joins. All current numbers are development-only and trace to the "
        "machine-readable bindings in state/v22_execution_state.json. Qualification and final evidence remain "
        f"governed by their manifests.\n\nCurrent development Core STC: {_metric(metrics, 'core_stc')}.\n",
        encoding="utf-8",
    )
    evidence_paths = [
        path
        for path in (
            ROOT / "state/v22_environment_manifest.json",
            ROOT / "state/v22_dataset_manifest.json",
            ROOT / "state/v22_model_bundle.json",
            ROOT / "state/v22_integrity.json",
            ROOT / "state/v22_review_findings.json",
            ROOT / "state/v22_qualification_manifest.json",
            ROOT / "state/v22_freeze_manifest.json",
            ROOT / "results/v22/v22_tests.xml",
        )
        if path.exists()
    ]
    resume = {
        "schema_version": 1,
        "created_at": utc_now(),
        "execution_status": execution.get("status"),
        "current_phase": execution.get("current_phase"),
        "artifacts": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in evidence_paths
        ],
    }
    atomic_write_json(ROOT / "artifacts/v22_resume_evidence.json", resume)
    (ROOT / "artifacts/v22_resume_evidence.md").write_text(
        "# V2.2 Resume Evidence\n\n"
        f"Status: {resume['execution_status']}; phase: {resume['current_phase']}.\n\n"
        + "\n".join(f"- `{item['path']}` `{item['sha256']}`" for item in resume["artifacts"])
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
