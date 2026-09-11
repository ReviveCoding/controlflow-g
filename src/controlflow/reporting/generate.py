from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from controlflow.core.state import PhaseRun, ProjectPaths, sha256_file, utc_now


def _read(paths: ProjectPaths, stem: str) -> pd.DataFrame:
    target = paths.root / "results" / f"{stem}.parquet"
    return pd.read_parquet(target) if target.exists() else pd.DataFrame()


def _metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "metrics" not in frame:
        return frame
    parsed = pd.DataFrame([json.loads(value) if isinstance(value, str) else {} for value in frame.metrics])
    return pd.concat([frame.drop(columns="metrics").reset_index(drop=True), parsed], axis=1)


def _save_bar(frame: pd.DataFrame, x: str, y: str, title: str, target: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 4.8))
    frame.plot.bar(x=x, y=y, ax=axis, legend=False, color="#245b78")
    axis.set_title(title)
    axis.set_ylabel(y.replace("_", " "))
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)


def _save_line(frame: pd.DataFrame, x: str, y: str, title: str, target: Path, group: str | None = None) -> None:
    figure, axis = plt.subplots(figsize=(9, 4.8))
    if group:
        for name, current in frame.groupby(group):
            axis.plot(current[x], current[y], marker="o", label=str(name))
        axis.legend()
    else:
        axis.plot(frame[x], frame[y], marker="o", color="#245b78")
    axis.set_title(title)
    axis.set_xlabel(x.replace("_", " "))
    axis.set_ylabel(y.replace("_", " "))
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)


def _diagram(target: Path, labels: list[str], title: str) -> None:
    figure, axis = plt.subplots(figsize=(12, 2.8))
    axis.axis("off")
    count = len(labels)
    for index, label in enumerate(labels):
        x = (index + 0.5) / count
        axis.text(
            x,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "#e8f1f5", "edgecolor": "#245b78"},
        )
        if index < count - 1:
            axis.annotate(
                "",
                xy=((index + 1.35) / count, 0.5),
                xytext=((index + 0.65) / count, 0.5),
                arrowprops={"arrowstyle": "->"},
            )
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)


def generate_figures(paths: ProjectPaths) -> list[Path]:
    output = paths.root / "reports/figures"
    output.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    def make(name: str, operation: Callable[[Path], None]) -> None:
        target = output / name
        operation(target)
        generated.append(target)

    make(
        "architecture.png",
        lambda target: _diagram(
            target,
            ["Identity", "PIT retrieval", "Risk + anomaly", "LLM", "Verifier", "Authorization", "HITL/action"],
            "ControlFlow-G governed execution",
        ),
    )
    make(
        "data_lineage.png",
        lambda target: _diagram(
            target, ["Official sources", "Raw", "Bronze", "Silver/SCD2", "Gold", "Features"], "Data lineage"
        ),
    )
    cases = pd.read_parquet(paths.root / "data/silver/synthetic_cases_development.parquet")
    make(
        "class_imbalance.png",
        lambda target: _save_bar(
            cases.severity.value_counts().rename_axis("severity").reset_index(name="count"),
            "severity",
            "count",
            "Development severity distribution",
            target,
        ),
    )
    drift = _read(paths, "drift")
    make("temporal_drift.png", lambda target: _save_bar(drift, "signal", "psi", "Observed drift signals", target))
    models = pd.concat([_metrics(_read(paths, "ml_models")), _metrics(_read(paths, "ml_models_advanced"))])
    make(
        "model_comparison.png",
        lambda target: _save_bar(models, "experiment_id", "macro_f1", "Validation model comparison", target),
    )
    calibration = _read(paths, "calibration")
    make(
        "calibration.png",
        lambda target: _save_bar(calibration, "method", "ece", "Validation calibration error", target),
    )
    risk = _read(paths, "risk_coverage")
    make(
        "risk_coverage.png",
        lambda target: _save_line(risk, "automation_coverage", "residual_critical_error", "Risk-coverage", target),
    )
    anomaly = _metrics(_read(paths, "anomaly"))
    make(
        "anomaly_comparison.png",
        lambda target: _save_bar(anomaly, "experiment_id", "pr_auc", "Anomaly validation", target),
    )
    semi = _metrics(_read(paths, "semi_supervised"))
    make(
        "semi_supervised.png",
        lambda target: _save_line(semi, "labeled_fraction", "macro_f1", "Label efficiency", target, "method"),
    )
    retrieval = _metrics(_read(paths, "retrieval"))
    make(
        "retrieval_comparison.png",
        lambda target: _save_bar(retrieval, "experiment_id", "recall_at_5", "Retrieval comparison", target),
    )
    chunking = _read(paths, "chunking")
    make(
        "chunking_comparison.png",
        lambda target: _save_bar(chunking, "strategy", "recall_at_5", "Chunking comparison", target),
    )
    retrieval_scale = _read(paths, "retrieval_scaling")
    make(
        "retrieval_scaling.png",
        lambda target: _save_line(retrieval_scale, "scale_chunks", "p95_latency_ms", "Retrieval scaling", target),
    )
    agents = _metrics(_read(paths, "agents"))
    make(
        "agent_comparison.png",
        lambda target: _save_bar(agents, "experiment_id", "nominal_task_success", "Agent nominal success", target),
    )
    make(
        "safe_task_completion.png",
        lambda target: _save_bar(agents, "experiment_id", "safe_task_completion", "Safe Task Completion", target),
    )
    security = _read(paths, "security")
    security_summary = security.groupby("system", as_index=False).blocked.mean()
    make(
        "security_results.png",
        lambda target: _save_bar(security_summary, "system", "blocked", "Adversarial attacks blocked", target),
    )
    hitl = _read(paths, "hitl")
    make(
        "review_residual_risk.png",
        lambda target: _save_line(
            hitl, "reviewer_error_rate", "residual_risk_mean", "Reviewer error sensitivity", target
        ),
    )
    ablation = _read(paths, "ablation")
    make(
        "ablation.png",
        lambda target: _save_bar(ablation, "ablation", "safe_task_completion", "Executed ablations", target),
    )
    make(
        "latency.png",
        lambda target: _save_bar(agents, "experiment_id", "p95_latency_seconds", "Agent P95 latency", target),
    )
    make(
        "cost.png",
        lambda target: _save_bar(agents, "experiment_id", "compute_cost_proxy_per_case", "Compute cost proxy", target),
    )
    scaling = _read(paths, "scaling")
    make(
        "spark_scaling.png",
        lambda target: _save_line(
            scaling, "scale", "rows_per_second", "Dataframe/Spark scaling", target, "implementation"
        ),
    )
    make(
        "pareto_frontier.png",
        lambda target: _save_line(
            agents.sort_values("p95_latency_seconds"),
            "p95_latency_seconds",
            "safe_task_completion",
            "Quality-latency frontier",
            target,
        ),
    )
    return generated


def _table(paths: ProjectPaths, stem: str, columns: list[str] | None = None) -> str:
    frame = _metrics(_read(paths, stem))
    if frame.empty:
        return "No executed artifact was available."
    if columns:
        frame = frame[[column for column in columns if column in frame.columns]]
    return "```text\n" + frame.to_string(index=False, max_rows=30) + "\n```"


def generate_documents(paths: ProjectPaths) -> list[Path]:
    final = _read(paths, "final_test")
    decision = str(final.iloc[0].release_decision) if not final.empty else "NOT_YET_EVALUATED"
    final_metrics = json.loads(final.iloc[0].metrics) if not final.empty else {}
    sections = {
        "03_data_engineering.md": (
            "Data Engineering",
            "PySpark/Delta-style medallion, SCD2, CDC and restart evidence.",
            "scaling",
        ),
        "04_ml_modeling.md": (
            "ML Modeling",
            "Validation-only supervised, calibration, anomaly and label-efficiency results.",
            "ml_models",
        ),
        "05_retrieval.md": (
            "Retrieval",
            "Independent queries, public corpora, GPU embeddings/reranking and safety prefilters.",
            "retrieval",
        ),
        "06_agent_architecture.md": (
            "Agent Architecture",
            "Observed AG0-AG6 execution with deterministic boundaries.",
            "agents",
        ),
        "07_security_governance.md": (
            "Security and Governance",
            "Fifteen executed adversarial control exercises.",
            "security",
        ),
        "08_reliability_operations.md": (
            "Reliability and Operations",
            "Injected timeout, crash, replay and checkpoint scenarios.",
            "operations",
        ),
        "09_ablation.md": (
            "Ablation Study",
            "Executed component removals on a paired validation scenario sample.",
            "ablation",
        ),
        "10_statistical_analysis.md": (
            "Statistical Analysis",
            "Validation comparisons use paired exact and bootstrap procedures.",
            "validation_statistics",
        ),
        "11_cost_scaling.md": (
            "Cost and Scaling",
            "Local measurements; cost is a compute proxy, not a cloud invoice.",
            "scaling",
        ),
        "12_final_evaluation.md": (
            "Locked Final Evaluation",
            "Generated only from the once-consumed holdout.",
            "final_test",
        ),
        "13_release_decision.md": (
            "Release Decision",
            f"Decision: {decision}. Gates were frozen before P26.",
            "final_test",
        ),
        "14_limitations.md": (
            "Limitations",
            "Synthetic enterprise truth, small local LLM, smoke-scale agents, local simulation, "
            "and partial whole-tree typing limit external validity.",
            "data_failures",
        ),
        "15_future_work.md": (
            "Future Work",
            "Use a new sealed holdout for any modified candidate; expand human review and "
            "deployment evidence without reusing this test.",
            "deployment",
        ),
    }
    written = []
    for filename, (title, narrative, stem) in sections.items():
        target = paths.root / "reports" / filename
        target.write_text(
            f"# {title}\n\n{narrative}\n\nArtifact: `results/{stem}.parquet`\n\n{_table(paths, stem)}\n",
            encoding="utf-8",
        )
        written.append(target)
    evidence = {
        "generated_at": utc_now(),
        "release_decision": decision,
        "final_metrics": final_metrics,
        "source_artifact": "results/final_test.parquet" if not final.empty else None,
        "source_sha256": sha256_file(paths.root / "results/final_test.parquet") if not final.empty else None,
    }
    cards = {
        "README.md": (
            "ControlFlow-G is a local, production-oriented research prototype-not a production deployment. "
            "See TECHNICAL_REPORT.md for executed evidence."
        ),
        "TECHNICAL_REPORT.md": (
            f"The governed candidate's frozen release decision is **{decision}**. "
            "All numbers trace to Parquet artifacts under `results/`."
        ),
        "MODEL_CARD.md": (
            "Risk models were trained on deterministic synthetic enterprise cases and evaluated on fixed "
            "validation splits. They are not fit for real financial decisions."
        ),
        "AGENT_CARD.md": (
            "The agent uses a pinned local Qwen model with deterministic retrieval, verification, authorization, "
            "HITL and idempotent simulated writes."
        ),
        "DATA_CARD.md": (
            "Sources are NIST, CFPB, eCFR/GovInfo, SEC and deterministic synthetic cases. "
            "Complaint volume is not population prevalence."
        ),
        "RISK_REGISTER.md": (
            "Principal risks: synthetic-to-real gap, small qualitative sample, policy simplification, "
            "untyped experimental boundaries, and simulation-only operations."
        ),
        "THREAT_MODEL.md": (
            "Trust boundaries cover retrieved content, typed tools, SQL AST validation, runtime identity policy, "
            "signed approvals, and the append-only action chain."
        ),
        "REPRODUCIBILITY.md": (
            "Use the pinned `uv.lock`, recorded Windows/WSL environments, phase entry points, hashes in `state/`, "
            "and never rerun P26 after consumption."
        ),
    }
    for filename, summary in cards.items():
        target = paths.root / filename
        target.write_text(
            f"# {filename.removesuffix('.md').replace('_', ' ')}\n\n{summary}\n\nRelease status: **{decision}**.\n",
            encoding="utf-8",
        )
        written.append(target)
    evidence_target = paths.root / "artifacts/resume_evidence.json"
    evidence_target.parent.mkdir(parents=True, exist_ok=True)
    evidence_target.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_target = paths.root / "artifacts/resume_evidence.md"
    markdown_target.write_text(
        "# Resume evidence\n\nNo business-impact claim is made. Frozen local evaluation metrics are recorded in "
        "`results/final_test.parquet` and `artifacts/resume_evidence.json`.\n",
        encoding="utf-8",
    )
    written.extend([evidence_target, markdown_target])
    return written


def run() -> str:
    paths = ProjectPaths.discover()
    with PhaseRun("P30", paths) as phase:
        figures = generate_figures(paths)
        documents = generate_documents(paths)
        for target in [*figures, *documents]:
            phase.register(target, "figure" if target.suffix == ".png" else "report")
    return str(paths.root / "TECHNICAL_REPORT.md")


if __name__ == "__main__":
    print(run())
