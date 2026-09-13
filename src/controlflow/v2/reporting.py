# ruff: noqa: E501
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from controlflow.core.state import ProjectPaths, atomic_write_json, sha256_file, utc_now


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    def value(item: Any) -> str:
        return f"{item:.4f}" if isinstance(item, float) else str(item)

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(value(item) for item in row) + " |"
        for row in frame[columns].itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _write(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def generate_v2_reports() -> tuple[Path, Path]:
    paths = ProjectPaths.discover()
    reports = paths.root / "reports/v2"
    result_dir = paths.root / "results/v2"
    tournament = pd.read_parquet(result_dir / "model_tournament_all.parquet")
    critical = pd.read_parquet(result_dir / "critical_classifier.parquet")
    hierarchy = pd.read_parquet(result_dir / "hierarchical_severity.parquet")
    fusion = pd.read_parquet(result_dir / "fusion.parquet")
    round4 = pd.read_parquet(result_dir / "selected_candidate_internal_eligibility_r4.parquet")
    r4 = round4.iloc[0]
    smoke = json.loads((paths.state / "v2_structured_output_smoke.json").read_text(encoding="utf-8"))
    quality = json.loads((paths.state / "v2_quality_checks.json").read_text(encoding="utf-8"))
    integrity = json.loads((paths.state / "v2_development_integrity.json").read_text(encoding="utf-8"))

    _write(
        reports / "01_model_tournament.md",
        "V2 Model Tournament",
        "Development-only screening results:\n\n"
        + _table(
            tournament,
            [
                "architecture",
                "cases",
                "structured_output_failure_rate",
                "critical_recall",
                "safe_task_completion",
                "p95_latency_seconds",
            ],
        )
        + "\n\nThe 32-case screen is architecture screening only; it is not eligibility or final evidence. Source: `results/v2/model_tournament_all.parquet`.",
    )
    _write(
        reports / "02_critical_classifier.md",
        "V2 Critical Classifier",
        _table(critical, ["model", "critical_recall", "auprc", "brier", "ece", "review_rate", "threshold"])
        + "\n\nThe deployed development candidate uses the weighted logistic route for robustness. Round-4 critical recall was "
        f"{r4['critical_recall']:.4f} at a {r4['critical_review_coverage']:.4f} critical-positive prediction rate. "
        "The latter artifact field is mislabeled as critical_review_coverage; it is not actual review coverage. Sources: "
        "`results/v2/critical_classifier.parquet` and `results/v2/selected_candidate_internal_eligibility_r4.parquet`.",
    )
    _write(
        reports / "03_structured_generation.md",
        "V2 Structured Generation",
        f"The pinned BF16 Qwen3-4B constrained smoke was syntactically valid: {smoke['constrained']['syntactically_valid']}. "
        f"Round 4 structured-output failure was {r4['structured_output_failure_rate']:.4f}; final assembled semantic failure "
        f"was {r4['semantic_failure_rate']:.4f}, while raw generated semantic failure was "
        f"{r4['generated_semantic_failure_rate']:.4f}. Deterministic typed fallback is therefore material, and raw generation "
        "quality must not be conflated with final packet validity. Sources: `state/v2_structured_output_smoke.json` and "
        "`results/v2/selected_candidate_internal_eligibility_r4.parquet`.",
    )
    _write(
        reports / "04_fusion.md",
        "V2 Fusion Study",
        _table(fusion, ["model", "critical_recall", "review_rate", "auprc", "brier", "ece"])
        + "\n\nSeverity/root fusion results:\n\n"
        + _table(hierarchy, ["model", "accuracy", "macro_f1", "critical_recall", "critical_ece", "review_coverage"])
        + "\n\nSources: `results/v2/fusion.parquet` and `results/v2/hierarchical_severity.parquet`.",
    )
    _write(
        reports / "05_qlora.md",
        "V2 QLoRA Decision",
        "QLoRA was not run. Although raw generated semantic failure remained measurable, the blocking round-4 failures were "
        "typed severity/action completion and approval bypasses, not JSON syntax. An adapter was not trained because it could "
        "not legitimately repair those deterministic/model-routing failures, and no final evaluation was authorized.",
    )
    _write(
        reports / "06_ablation.md",
        "V2 Ablation Status",
        "A release-grade controlled ablation was not run. The earlier `results/v2/ablation.parquet` contains development "
        "scenario diagnostics and counterfactual counts, not causal one-factor ablations. Round 4 remained below the STC gate, "
        "so spending full runtime on ablations would violate the phase rule.",
    )
    _write(
        reports / "07_final_evaluation.md",
        "V2 Final Evaluation",
        "NOT RUN. No fresh V2 final holdout was created or consumed because development eligibility failed. The V1 333-case "
        "holdout was not used. There are no V2 final metrics.",
    )
    _write(
        reports / "08_release_decision.md",
        "V2 Release Decision",
        "NO_PROMOTE\n\nStatus: `V2 DEVELOPMENT_NO_GO`. Round 4 failed Safe Task Completion and approval-bypass gates; "
        "the independent reviews also found the stale-policy eligibility fixture vacuous and did not clear security. "
        "No final holdout, freeze, or one-shot final was authorized.",
    )

    technical = paths.root / "TECHNICAL_REPORT_V2.md"
    _write(
        technical,
        "ControlFlow-G V2 Technical Report",
        "ControlFlow-G V2 preserved the V1 reference boundary and rebuilt generation as JSON-Schema-constrained Qwen3-4B "
        "inference with deterministic typed assembly, independent point-in-time retrieval, external authorization, dedicated "
        "critical-risk routing, root-cause prediction, and severity fusion.\n\n"
        f"The decisive 324-case internal eligibility round produced binary critical-risk recall {r4['critical_recall']:.4f}, STC "
        f"{r4['safe_task_completion']:.4f}, structured-output failure {r4['structured_output_failure_rate']:.4f}, "
        f"unauthorized irreversible simulated actions {int(r4['unauthorized_irreversible_actions'])}, approval bypasses "
        f"{int(r4['approval_bypasses'])}, stale-policy error {r4['stale_policy_error_rate']:.4f}, and P95 latency "
        f"{r4['p95_latency_seconds']:.4f}s. These values come from "
        "`results/v2/selected_candidate_internal_eligibility_r4.parquet`.\n\n"
        "Typed severity recall on the 99 critical cases was 74/99 (0.7475), distinct from binary critical-risk recall. "
        "The reported critical_review_coverage field is actually the overall critical-positive prediction rate. Independent "
        "review also found that the all-2025 eligibility fixture did not meaningfully test historical stale-policy selection.\n\n"
        "The development decision is NO_PROMOTE. V2-16 through V2-18 were not executed. V1 remains historical diagnostic "
        "evidence only.",
    )

    source_paths = [
        paths.state / "v2_development_eligibility.json",
        paths.state / "v2_development_integrity.json",
        paths.state / "v2_quality_checks.json",
        result_dir / "selected_candidate_internal_eligibility_r4.parquet",
        result_dir / "selected_candidate_internal_eligibility_r4_traces.parquet",
        paths.state / "v2_internal_eligibility_protocol_r4.json",
    ]
    resume: dict[str, Any] = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "V2_DEVELOPMENT_NO_GO",
        "release_decision": "NO_PROMOTE",
        "development_eligible": False,
        "fresh_v2_final_holdout_created": False,
        "v2_final_consumed": False,
        "v1_final_reused": False,
        "quality_checks_passed": bool(quality["passed"]),
        "integrity_checks_passed": bool(integrity["passed"]),
        "round4_metrics": {
            key: (value.item() if hasattr(value, "item") else value) for key, value in r4.to_dict().items()
        },
        "artifacts": [
            {"path": str(path.relative_to(paths.root)), "sha256": sha256_file(path)} for path in source_paths
        ],
        "resume_instruction": "Do not create a final holdout. Begin a new development iteration with new internal eligibility data.",
    }
    resume_json = paths.root / "artifacts/v2_resume_evidence.json"
    atomic_write_json(resume_json, resume)
    resume_md = paths.root / "artifacts/v2_resume_evidence.md"
    _write(
        resume_md,
        "V2 Resume Evidence",
        "Status: **V2 DEVELOPMENT_NO_GO**  \nRelease decision: **NO_PROMOTE**  \nFresh V2 final holdout created: **No**  "
        "\nV2 final consumed: **No**  \nV1 final reused: **No**\n\nResume only as a new development iteration; do not create a final holdout from this state.",
    )
    return technical, resume_json
