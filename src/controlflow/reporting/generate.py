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


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    rate = successes / total
    denominator = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / denominator
    radius = z * ((rate * (1 - rate) / total + z**2 / (4 * total**2)) ** 0.5) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


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


def _save_histogram(frame: pd.DataFrame, column: str, title: str, target: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.hist(frame[column], bins=40, color="#245b78")
    axis.set_title(title)
    axis.set_xlabel(column.replace("_", " "))
    axis.set_ylabel("count")
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
    make(
        "data_distributions.png",
        lambda target: _save_histogram(cases, "amount", "Case amount distribution", target),
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
    calibration_bins = _read(paths, "calibration_bins")
    make(
        "reliability_curve.png",
        lambda target: _save_line(
            calibration_bins,
            "mean_confidence",
            "observed_accuracy",
            "Validation reliability curves",
            target,
            "method",
        ),
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
    return "```text\n" + str(frame.to_string(index=False, max_rows=30)) + "\n```"


def generate_documents(paths: ProjectPaths) -> list[Path]:
    final = _read(paths, "final_test")
    decision = str(final.iloc[0].release_decision) if not final.empty else "NOT_YET_EVALUATED"
    final_metrics = json.loads(final.iloc[0].metrics) if not final.empty else {}
    final_statistics = _read(paths, "final_statistics")
    final_stat_metrics = json.loads(final_statistics.iloc[0].metrics) if not final_statistics.empty else {}
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
            "Observed latency plus three explicit review/error-cost sensitivity scenarios. Cost is a unitless "
            "proxy, not a cloud invoice or business-impact estimate. Spark measurements remain in "
            "results/scaling.parquet.",
            "business_metrics",
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
            "bounded retrieval scale, and limited deep-model repeats constrain external validity. The full source tree "
            "passes strict static typing, but that does not establish semantic correctness.",
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
    claims: list[dict[str, object]] = []
    if final_metrics and final_stat_metrics:
        claims.append(
            {
                "claim": (
                    "In the one-time locked synthetic holdout, ControlFlow-G achieved a measured Safe Task "
                    "Completion rate compared with the unrestricted ReAct baseline."
                ),
                "baseline": final_stat_metrics["baseline_stc"],
                "measured_result": final_stat_metrics["candidate_stc"],
                "effect": final_stat_metrics["paired_stc_difference"],
                "experiment_id": "final-paired-AG6-vs-AG3",
                "dataset": "locked deterministic synthetic enterprise holdout",
                "sample_size": final_stat_metrics["sample_size"],
                "confidence_interval": [
                    final_stat_metrics["paired_bootstrap_ci95_low"],
                    final_stat_metrics["paired_bootstrap_ci95_high"],
                ],
                "artifact_path": "results/final_statistics.parquet",
            }
        )
    security = _read(paths, "security")
    if not security.empty:
        claims.append(
            {
                "claim": "The validation adversarial suite executed all specified S01-S15 scenarios.",
                "baseline": "not applicable",
                "measured_result": float(security.blocked.mean()),
                "experiment_id": "security-AG6-S01..S15",
                "dataset": "deterministic adversarial validation suite",
                "sample_size": len(security),
                "confidence_interval": _wilson_interval(int(security.blocked.sum()), len(security)),
                "artifact_path": "results/security.parquet",
            }
        )
    operations = _read(paths, "operations")
    injected = operations[operations.injected] if not operations.empty else operations
    if not injected.empty:
        claims.append(
            {
                "claim": "Injected validation failure scenarios recovered without duplicate simulated execution.",
                "baseline": "not applicable",
                "measured_result": float(injected.recovery_success.mean()),
                "experiment_id": "reliability-00..10",
                "dataset": "local fault-injection validation suite",
                "sample_size": len(injected),
                "confidence_interval": _wilson_interval(int(injected.recovery_success.sum()), len(injected)),
                "artifact_path": "results/operations.parquet",
            }
        )
    evidence = {
        "generated_at": utc_now(),
        "release_decision": decision,
        "final_metrics": final_metrics,
        "claims": claims,
        "source_artifact": "results/final_test.parquet" if not final.empty else None,
        "source_sha256": sha256_file(paths.root / "results/final_test.parquet") if not final.empty else None,
    }
    cards = {
        "README.md": (
            "ControlFlow-G is a governed, production-oriented research prototype for control-exception "
            "investigation. It combines a local Delta/Spark lakehouse, point-in-time features, calibrated risk "
            "and anomaly models, temporal hybrid retrieval, typed tools, deterministic authorization, evidence "
            "verification, HITL, and idempotent simulated actions. It is not a bank system or production "
            "deployment.\n\n"
            f"## Frozen result\n\nRelease decision: **{decision}**. Final metrics are machine-readable in "
            "`results/final_test.parquet`; paired AG6-versus-AG3 inference is in "
            "`results/final_statistics.parquet`. Negative and null outcomes are retained.\n\n"
            "## Reproduce\n\nUse Python 3.11 and the pinned `uv.lock`: `uv sync --extra dev`, then "
            "`uv run pytest -q`, `uv run ruff check src tests scripts`, and `uv run mypy src`. Public acquisition "
            "is exposed by `python -m controlflow.data.download --help`. Spark/Delta phases use the recorded WSL "
            "runtime; CUDA phases fail rather than silently fall back. Never invoke P26 after the seal is consumed.\n\n"
            "## Evidence map\n\nSee `TECHNICAL_REPORT.md`, `REPRODUCIBILITY.md`, reports 00-15, state manifests, "
            "and Parquet result tables. Generated figures are under `reports/figures/`."
        ),
        "TECHNICAL_REPORT.md": (
            f"## Research question\n\nThe study tests whether governed data, model, retrieval, authorization, and "
            "execution boundaries improve Safe Task Completion over less-constrained agents without assuming a "
            f"positive result. The frozen decision is **{decision}**.\n\n"
            "## Executed design\n\nPublic NIST, CFPB, Title 12 CFR, and bounded SEC data were acquired and hashed. "
            "Deterministic synthetic enterprise truth supplies authorization/action labels. Validation includes "
            "supervised, anomaly, semi-supervised, calibration, retrieval, agent, security, recovery, scaling, "
            "ablation, interaction, and paired statistical studies. The final holdout was unlocked once only after "
            "a clean-tree freeze.\n\n"
            "## Architecture\n\nIdentity and purpose precede pre-search authorization. Evidence and features are "
            "point-in-time constrained. The LLM cannot write SQL or authorize itself; typed actions require policy, "
            "verification, and where necessary a provisioned signed review decision. Action events are idempotent, "
            "hash chained, HMAC anchored, and rollback-capable in simulation.\n\n"
            "## Result provenance\n\nEvery quantitative statement is derived from `results/` artifacts and registered "
            "hashes. The exact final gates are in `configs/release_gates.yaml`; final metrics follow.\n\n"
            f"{json.dumps(final_metrics, indent=2, sort_keys=True)}\n\n"
            "## Interpretation\n\nThis is evidence about a deterministic public-data/synthetic benchmark on one "
            "workstation, not evidence of banking production impact. See reports 03-15 for subsystem tables and "
            "`reports/14_limitations.md` for validity constraints."
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
            "bounded local scale, limited deep repeats, and simulation-only operations."
        ),
        "THREAT_MODEL.md": (
            "Trust boundaries cover retrieved content, typed tools, SQL AST validation, runtime identity policy, "
            "signed approvals, and the append-only action chain."
        ),
        "REPRODUCIBILITY.md": (
            "## Environment\n\nUse the pinned `uv.lock` and manifests in `state/`. Windows executes CUDA ML; "
            "Ubuntu 22.04 WSL2 executes Spark 3.5.9/Delta 3.3.2 against the same repository.\n\n"
            "## Validation\n\nRun `uv run ruff format --check src tests scripts`, `uv run ruff check src tests "
            "scripts`, `uv run mypy src`, and `uv run pytest -q`. Verify artifact hashes with "
            "`controlflow.core.state.verify_artifact_manifest()`.\n\n"
            "## Scientific lock\n\nDataset and split hashes are in `state/data_manifest.json` and "
            "`state/split_manifest.json`; the candidate identity is in `state/freeze_manifest.json`. P26 is "
            "fail-closed and must never be rerun after consumption. A material post-final bug requires a new holdout."
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
    claim_lines = [
        f"- {item['claim']} Baseline: {item['baseline']}; result: {item['measured_result']}; "
        f"experiment: `{item['experiment_id']}`; n={item['sample_size']}; 95% CI: {item['confidence_interval']}; "
        f"artifact: `{item['artifact_path']}`."
        for item in claims
    ]
    markdown_target.write_text(
        "# Resume evidence\n\nThis is a public-data, deterministic-synthetic, production-like simulation. "
        "No banking production or business-impact claim is made.\n\n" + "\n".join(claim_lines) + "\n",
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
