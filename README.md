# ControlFlow-G V2.6

ControlFlow-G studies how a local agent can investigate synthetic financial-services control exceptions while keeping risk decisions, authorization, actions, and evaluation independently verifiable.

## Status

**Synthetic research release: PROMOTE under the frozen V2.6 evaluation protocol.** The research release is the immutable [`controlflow-g-v26-promote`](https://github.com/ReviveCoding/controlflow-g/tree/controlflow-g-v26-promote) tag. This is a local, production-like simulation, not a bank or production deployment. It performs no real financial actions.

## Problem and architecture

An agent can retrieve the wrong policy version, miss a critical case, trust adversarial evidence, or propose an action outside its authority. ControlFlow-G separates learned risk estimates from deterministic policy and action control. Point-in-time data and policy retrieval, calibrated models, bounded read-only tools, evidence verification, external runtime authorization, human approval, idempotent simulated actions, and a tamper-evident audit trail form one evaluated workflow.

`synthetic/public-data lakehouse → calibrated models → temporal retrieval → bounded agent → evidence verifier → PDP/PEP → approval → transactional simulated action → audit → frozen release gates`

The LLM may produce a structured explanation; it does not define ground truth or authorize a commit. See [TECHNICAL_REPORT_V26.md](TECHNICAL_REPORT_V26.md) for the design and threat model.

## Frozen V26FINAL result

The one-shot, sealed 600-case final run used the frozen Qwen3/vLLM configuration at observed maximum LLM concurrency 2. The table comes from [`results/v26/final/gate_decision.json`](results/v26/final/gate_decision.json), [`results/v26/final/denominators.json`](results/v26/final/denominators.json), and [`state/v26_release_decision.json`](state/v26_release_decision.json).

| Measure | Final result |
| --- | ---: |
| Core STC | 489/600 (81.50%) |
| Binary critical recall | 141/144 (97.92%) |
| Typed CRITICAL recall | 141/144 (97.92%) |
| Stale-policy errors | 0/150 |
| Structured-output failures | 0/600 |
| Unauthorized commits | 0/83 |
| Approval-bypass commits | 0/255 |
| Duplicate commits | 0/600 |
| Temporal policy accuracy | 600/600 (100%) |
| Evidence completeness | 1005/1005 (100%) |
| Total P95 latency, concurrency 2 | 7.392 seconds |
| Frozen release gates | 17/17 passed |

Zero observed failures in this synthetic sample do not establish zero true risk.

## Evaluation and reliability lessons

Development runs and earlier failed qualifications are diagnostic, not final evidence. Each failed qualification identity was permanently consumed. Structural fixes preceded a new holdout, a fresh V26QUAL, independent post-qualification review, and a fresh one-shot V26FINAL. The release decision and artifact hashes are in [`state/v26_release_decision.json`](state/v26_release_decision.json) and [`reports/v26/release.md`](reports/v26/release.md).

Iterations V2.1–V2.6 hardened truth separation and authorization, serving latency, zero-denominator evaluation, SQLite lifecycle, checkpoint schema, semantic-versus-provenance comparison, and cross-platform frozen-byte portability. Those defects and consumed runs are documented in the versioned reports; they are part of the method's audit trail.

## Reproducibility and limits

[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) gives the portable checks. [`scripts/verify_release_portability.py`](scripts/verify_release_portability.py) checks the immutable promoted tag against the additive line-ending attestation and original frozen receipts; [GitHub Actions](https://github.com/ReviveCoding/controlflow-g/actions) runs those checks on the post-release branch. The research release remains at `f21a4192373c5bcb815e12a7fa10ac8f183b482b`. The separate `controlflow-g-v26-public` tag identifies the MIT-licensed public package with later portability and documentation work; it does not change the frozen result.

The study uses public-data inputs and synthetic cases with deterministic evaluator truth. Local GPU evidence concerns an NVIDIA RTX 4090 Laptop GPU; it does not establish other hardware performance or bank production fitness. No real financial actions are available.

## License

ControlFlow-G is licensed under the MIT License. [LICENSE](LICENSE) contains the authoritative distribution terms and matches the existing MIT package metadata.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/controlflow/` | Data, models, retrieval, agent runtime, authorization, audit, and evaluation code |
| `configs/` | Resource limits, schemas, versioned policies, denominators, and gates |
| `data/`, `artifacts/`, `results/` | Synthetic cohorts, execution artifacts, and scored evidence |
| `state/` | Freeze manifests, execution state, review findings, receipts, and release decision |
| `reports/` | Versioned research reports and portability attestation |
| `scripts/` | Reproducible checks and phase entry points |
| `tests/` | Portable and specialized test suites |

See [PROJECT_SPEC.md](PROJECT_SPEC.md) for the study protocol, [RELEASE_NOTES_V26.md](RELEASE_NOTES_V26.md) for the local release-notes draft, and [SECURITY.md](SECURITY.md) for security reporting.

