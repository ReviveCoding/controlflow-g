from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from controlflow.core.state import atomic_write_json, sha256_file, utc_now
from controlflow.v22.approval import ApprovalIssuer
from controlflow.v22.bundle import write_bundle
from controlflow.v22.candidate import CandidateExecutionWorkflow, CandidateModelBundle
from controlflow.v22.evaluation import evaluate
from controlflow.v22.runtime import build_candidate_runtime
from controlflow.v22.schemas import PolicyDecision

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/v22/runtime.yaml").read_text(encoding="utf-8"))
    namespace = config["development"]["namespace"]
    base = ROOT / f"data/v22/{namespace}/validation"
    runtime = pd.read_parquet(base / "runtime_cases.parquet")
    truth = base / "evaluator_truth.parquet"
    bundle = CandidateModelBundle(ROOT / "state/v22_model_bundle.json", ROOT)
    public_key = ROOT / "artifacts/v22/approval_public_key.pem"
    issuer = ApprovalIssuer.create_ephemeral(ROOT / "artifacts/v22/local_keys", public_key)
    definitions: dict[str, dict[str, bool]] = {
        "FULL": {},
        "NO_HARD_CRITICAL_ROUTING": {"hard_critical_routing": False},
        "NO_MANDATORY_REVIEW": {},
        "NO_TEMPORAL_FILTER": {"temporal_filter": False},
        "NO_UNCERTAINTY_ROUTING": {"uncertainty_routing": False},
    }
    summaries: dict[str, Any] = {}
    per_case: dict[str, pd.DataFrame] = {}
    ablation_payload = {
        key: value
        for key, value in bundle.payload.items()
        if key not in {"schema_version", "created_at", "bundle_sha256"}
    }
    ablation_payload["policy_config"] = {
        "path": "configs/v22/policy_no_mandatory_review.yaml",
        "sha256": sha256_file(ROOT / "configs/v22/policy_no_mandatory_review.yaml"),
    }
    ablation_payload["action_registry"] = {
        "path": "configs/v22/action_registry_no_mandatory_review.yaml",
        "sha256": sha256_file(ROOT / "configs/v22/action_registry_no_mandatory_review.yaml"),
    }
    alternate_bundle_path = ROOT / f"results/v22/{namespace}_no_mandatory_review_bundle.json"
    write_bundle(alternate_bundle_path, ablation_payload)
    for name, options in definitions.items():
        slug = name.casefold()
        ledger = ROOT / f"artifacts/v22/{namespace}/ablation_{slug}.sqlite"
        if ledger.exists():
            raise RuntimeError(f"ablation evidence namespace already exists: {ledger}")
        selected_bundle_path = (
            alternate_bundle_path if name == "NO_MANDATORY_REVIEW" else ROOT / "state/v22_model_bundle.json"
        )
        candidate, _executor, selected_bundle = build_candidate_runtime(
            root=ROOT,
            bundle_path=selected_bundle_path,
            evidence_path=base / "evidence_corpus.parquet",
            authorization_path=base / "authorization_state.parquet",
            ledger_path=ledger,
        )
        runner = CandidateExecutionWorkflow(
            candidate,
            reviewer=None
            if name == "NO_MANDATORY_REVIEW"
            else lambda prepared: (
                issuer.issue(
                    action=prepared.action,
                    policy=prepared.proposal,
                    reviewer_id="external-reviewer-ablation",
                    approve=True,
                )
                if prepared.proposal.decision is PolicyDecision.REQUIRE_REVIEW
                else None
            ),
        )
        output = ROOT / f"results/v22/{namespace}_ablation_{slug}.parquet"
        pd.DataFrame(
            [runner.run_case(row, **options).model_dump(mode="json") for row in runtime.to_dict(orient="records")]
        ).to_parquet(output, index=False)
        metrics_path = ROOT / f"results/v22/{namespace}_ablation_{slug}_metrics.json"
        summaries[name] = evaluate(
            output,
            truth,
            ledger,
            metrics_path,
            critical_threshold=selected_bundle.threshold,
        )
        per_case[name] = pd.read_parquet(metrics_path.with_suffix(".per_case.parquet"))
    baseline = per_case["FULL"].set_index("case_id").core_stc_pass.astype(int)
    paired = {}
    for name in definitions:
        if name == "FULL":
            continue
        intervention = per_case[name].set_index("case_id").core_stc_pass.astype(int)
        difference = baseline - intervention
        paired[name] = {
            "cases": len(difference),
            "full_better": int((difference > 0).sum()),
            "intervention_better": int((difference < 0).sum()),
            "ties": int((difference == 0).sum()),
            "mean_core_stc_difference": float(difference.mean()),
        }
    structured_path = ROOT / "state/v22_vllm_structured_c2.json"
    unconstrained_path = ROOT / "state/v22_vllm_unconstrained_c2.json"
    semantic_path = ROOT / "state/v22_vllm_no_semantic_c2.json"
    llm_ablations = {
        "NO_CONSTRAINED_OUTPUT": json.loads(unconstrained_path.read_text()) if unconstrained_path.exists() else None,
        "NO_SEMANTIC_VERIFIER": json.loads(semantic_path.read_text()) if semantic_path.exists() else None,
        "FULL_STRUCTURED_SERVING": json.loads(structured_path.read_text()) if structured_path.exists() else None,
    }
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "role": "DEVELOPMENT_VALIDATION",
        "interventions_reran_relevant_components": all(
            (ROOT / f"results/v22/{namespace}_ablation_{name.casefold()}.parquet").is_file() for name in definitions
        ),
        "summaries": summaries,
        "paired_core_stc": paired,
        "llm_ablations": llm_ablations,
        "qualification_evidence": False,
    }
    atomic_write_json(ROOT / "results/v22/development_ablations.json", report)


if __name__ == "__main__":
    main()
