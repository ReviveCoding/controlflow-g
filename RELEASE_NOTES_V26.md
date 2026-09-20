# ControlFlow-G V2.6 — Governed Agent Reliability Research Release

The frozen research evaluation is `controlflow-g-v26-promote` at commit `f21a4192373c5bcb815e12a7fa10ac8f183b482b`. The public distribution snapshot is the separate annotated `controlflow-g-v26-public` tag at its peeled packaging commit. The release decision is **PROMOTE for a synthetic research artifact**, not for bank or production deployment. The public tag adds MIT licensing, documentation, and clean-clone CI portability after the frozen result; it does not change the evaluation or metrics.

## Scope and architecture

ControlFlow-G combines a local lakehouse, point-in-time features, calibrated risk models, temporal and authorization-filtered retrieval, a bounded structured agent, evidence verification, external PDP/PEP authorization, human approval, transactional idempotent simulated actions, and a tamper-evident audit chain. Evaluator truth remains separate from runtime inputs. No real financial actions are performed.

## Frozen V26FINAL result

| Measure | Result |
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
| Total P95 latency at concurrency 2 | 7.392 seconds |
| Frozen gates | 17/17 passed |

These values are from `results/v26/final/gate_decision.json`, `results/v26/final/denominators.json`, and `state/v26_release_decision.json` at the promoted tag.

## Evaluation and integrity

Development runs did not substitute for final evidence. Failed qualification identities were consumed. Structural fixes preceded a fresh V26QUAL, independent post-qualification reviews, and a fresh one-shot V26FINAL. Frozen manifests, SHA-256 bindings, independent denominator checks, checkpoint validation, SQLite post-close checks, closure receipts, and terminal verification preserve the result's provenance. See `reports/v26/release.md` and `TECHNICAL_REPORT_V26.md` for detail.

## Portability and limitations

Post-release CI and public-readiness improvements live on `v2.6-development` after the frozen research-release tag. An additive line-ending attestation and `scripts/verify_release_portability.py` prove how normalized Git blobs reconstruct the original Windows frozen bytes and receipts; neither changes the promoted tag or original evidence. The `controlflow-g-v26-public` tag contains this release-notes source, the technical report, the reproducibility guide, and [LICENSE](https://github.com/ReviveCoding/controlflow-g/blob/controlflow-g-v26-public/LICENSE).

This study uses synthetic cases and public-data inputs, with local NVIDIA RTX 4090 Laptop GPU evidence. It is not a bank system or real financial-action workflow. Zero observed failures do not establish zero true risk or production suitability. The public package is licensed under MIT; [LICENSE](https://github.com/ReviveCoding/controlflow-g/blob/controlflow-g-v26-public/LICENSE) is the authoritative distribution text and aligns with `pyproject.toml` metadata.
