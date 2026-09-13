from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from controlflow.core.state import atomic_write_json, sha256_file, utc_now

ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict[str, Any] | None:
    target = ROOT / path
    return json.loads(target.read_text(encoding="utf-8")) if target.is_file() else None


def write(name: str, title: str, body: str) -> None:
    path = ROOT / "reports/v23" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.strip()}\n", encoding="utf-8")


def config_line(item: dict[str, Any]) -> str:
    latency = item["total_p95_seconds"]
    return (
        f"- `{item['config_id']}`: n={item['requests']}, total P95={latency:.3f}s, "
        f"structured failure={item['structured_failure_rate']:.4f}, Core STC={item['core_stc']:.4f}."
    )


def main() -> None:
    tournament = load("state/v23_serving_tournament.json") or {}
    attribution = load("state/v23_latency_attribution.json") or {}
    tail = load("results/v23/tail_analysis.json") or {}
    historical = load("results/v23/v22qual_structured_failure_diagnostics.json") or {}
    reviews = load("state/v23_review_findings.json") or {}
    qualification = load("state/v23_qualification_manifest.json") or {}
    freeze = load("state/v23_freeze_manifest.json") or {}
    final = load("results/v23/final_metrics.json")
    configs = tournament.get("configurations", [])
    lines = "\n".join(config_line(item) for item in configs) or "No completed benchmark summaries."
    baseline = next((item for item in configs if item["config_id"] == "baseline_v22_cold_60_r2"), None)
    valid_minimal = next(
        (item for item in configs if item["config_id"] == "p0_balanced_minimal_enum_128_b2048_warm_60"), None
    )
    write(
        "01_latency_attribution.md",
        "V2.3 Latency Attribution",
        "Request-level timestamps, token counts, per-request vLLM timings, Prometheus snapshots, and one-second "
        "NVIDIA samples are retained in the paths bound by `state/v23_latency_attribution.json`. "
        f"Selected analysis artifact: `{attribution.get('tail_analysis', {}).get('path', 'not yet selected')}`. "
        "No thermal-throttling claim is made without aligned temperature, power, and clock evidence.",
    )
    write(
        "02_server_tournament.md",
        "V2.3 Server Tournament",
        lines
        + "\n\nThe exact V2.2 server is labeled baseline. V2.3 candidates explicitly pin runtime arguments and use "
        "fixed development-only warm-up inputs.",
    )
    output_body = "E0 and E1 have not both completed."
    if baseline and valid_minimal:
        output_body = (
            f"The exact V2.2 current-schema cold baseline measured {baseline['total_p95_seconds']:.3f}s total P95. "
            f"The semantically verified minimal enum contract measured {valid_minimal['total_p95_seconds']:.3f}s "
            f"with structured failure {valid_minimal['structured_failure_rate']:.4f}. Deterministic safety fields "
            "remain assembled by the typed pipeline; the LLM returns only the summary and constrained evidence count."
        )
    write("03_output_contract.md", "V2.3 Output Contract", output_body)
    write(
        "04_scheduler_tuning.md",
        "V2.3 Scheduler Tuning",
        "The tournament records balanced/interactivity and 1024/2048/4096 batched-token experiments. "
        "Selection uses sustained end-to-end P95, structured validity, and unchanged safety results, "
        "not throughput alone.\n\n" + lines,
    )
    prefix = attribution.get("vllm_metrics", {})
    write(
        "05_prefix_cache.md",
        "V2.3 Prefix Cache",
        f"Selected-run Prometheus delta analysis: `{json.dumps(prefix, sort_keys=True)}`. Cold and warm runs are "
        "labeled separately; fixed warm-up inputs exclude qualification and final cases.",
    )
    write(
        "06_speculative_decoding.md",
        "V2.3 Speculative Decoding",
        "Not used. The primary structured-output optimization produced development latency headroom, so the optional "
        "speculative-decoding experiment was not justified.",
    )
    write(
        "07_tail_analysis.md",
        "V2.3 Tail Analysis",
        "Machine-readable top 1%, 5%, and 10% groups, correlations, category counts, telemetry, and vLLM deltas are "
        f"in `results/v23/tail_analysis.json`. Current winner: `{tail.get('winner', 'not selected')}`.",
    )
    counts: dict[str, int] = {}
    for item in historical.get("failures", []):
        mechanism = str(item["mechanism"])
        counts[mechanism] = counts.get(mechanism, 0) + 1
    write(
        "08_structured_failures.md",
        "V2.3 Structured Failure Analysis",
        "V22QUAL is consumed and diagnostic-only. Its six failures classify as "
        f"`{json.dumps(counts, sort_keys=True)}`; "
        "the unresolved fallback-shaped case is not assigned a fabricated mechanism. V2.3 candidate failure counts "
        "are reported per tournament configuration.",
    )
    write(
        "09_ablations.md",
        "V2.3 Paired Serving Ablations",
        "Completed configuration summaries below bind the paired development serving evidence.\n\n" + lines,
    )
    write(
        "10_review_disposition.md",
        "V2.3 Review Disposition",
        f"Review status: `{reviews.get('status', 'NOT_RUN')}`; unresolved counts: "
        f"`{json.dumps(reviews.get('counts', {}), sort_keys=True)}`.",
    )
    write(
        "11_internal_qualification.md",
        "V2.3 Internal Qualification",
        f"Status: `{qualification.get('status', 'NOT_RUN')}`. V23QUAL is a fresh one-shot protocol and is not "
        "generated before pre-qualification review clearance.",
    )
    write(
        "12_final_evaluation.md",
        "V2.3 Final Evaluation",
        "Final evaluation has not run."
        if final is None
        else f"Frozen final metrics: `{json.dumps(final, sort_keys=True)}`.",
    )
    decision = qualification.get("release_decision") or freeze.get("release_decision") or "NOT_REACHED"
    write(
        "13_release_decision.md",
        "V2.3 Release Decision",
        f"{decision}\n\nNo promotion decision is issued before qualification, post-qualification review, freeze, "
        "and one-shot final.",
    )
    write(
        "14_limitations.md",
        "V2.3 Limitations",
        "This is a local production-like simulation, not a bank deployment, and it performs no real financial action. "
        "Synthetic labels do not establish production validity. A local hash chain depends on trusted external "
        "anchoring. Laptop GPU thermals and host contention may limit transferability. Zero observed security failures "
        "do not prove zero risk.",
    )
    write(
        "15_future_work.md",
        "V2.3 Future Work",
        "Externalize signing keys and audit anchors, broaden primary-source grounding, repeat hardware "
        "characterization on deployment-class accelerators, and add independent human evaluation of explanation "
        "usefulness.",
    )
    technical = ROOT / "TECHNICAL_REPORT_V23.md"
    technical.write_text(
        "# ControlFlow-G V2.3 Technical Report\n\n"
        "V2.3 preserves the exact V2.2 typed decision and governance core while evaluating low-latency governed "
        "serving. "
        f"Tournament status: `{tournament.get('status', 'NOT_RUN')}`. Qualification status: "
        f"`{qualification.get('status', 'NOT_RUN')}`. Only machine-readable artifacts support quantitative claims.\n",
        encoding="utf-8",
    )
    evidence_paths = [
        path
        for path in (
            ROOT / "state/v23_historical_boundary.json",
            ROOT / "state/v23_environment_manifest.json",
            ROOT / "state/v23_typed_core_manifest.json",
            ROOT / "state/v23_serving_tournament.json",
            ROOT / "state/v23_latency_attribution.json",
            ROOT / "state/v23_review_findings.json",
            ROOT / "state/v23_qualification_manifest.json",
            ROOT / "state/v23_freeze_manifest.json",
        )
        if path.is_file()
    ]
    resume = {
        "schema_version": 1,
        "created_at": utc_now(),
        "artifacts": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in evidence_paths
        ],
    }
    atomic_write_json(ROOT / "artifacts/v23_resume_evidence.json", resume)
    (ROOT / "artifacts/v23_resume_evidence.md").write_text(
        "# V2.3 Resume Evidence\n\n"
        + "\n".join(f"- `{item['path']}` `{item['sha256']}`" for item in resume["artifacts"])
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
