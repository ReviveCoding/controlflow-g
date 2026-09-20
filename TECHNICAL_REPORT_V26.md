# ControlFlow-G V2.6 technical report

## Executive summary

ControlFlow-G is a local research system for governed investigation of synthetic financial-services control exceptions. The V2.6 release decision is **PROMOTE for a synthetic research artifact only**, recorded in [`state/v26_release_decision.json`](state/v26_release_decision.json). It is neither a bank deployment nor an authorization for real financial action. A fresh qualification and one-shot sealed final run completed the frozen protocol. V26FINAL passed all 17 predeclared gates; the result table below is transcribed from [`results/v26/final/gate_decision.json`](results/v26/final/gate_decision.json), [`results/v26/final/denominators.json`](results/v26/final/denominators.json), and the release decision.

## Problem definition and failure model

The task is to turn an exception record and its accessible evidence into a risk classification, policy-grounded explanation, and simulated disposition. A successful typed case requires the right criticality, evidence, temporal policy, and governed action rather than a plausible narrative alone. The study addresses label leakage into runtime inputs, missed critical cases, stale or future policy retrieval, prompt injection through evidence, unsafe tool use, unauthorized actions, approval replay or bypass, duplicate commits, and unverifiable evaluation artifacts. The phase contract is in [`PROJECT_SPEC.md`](PROJECT_SPEC.md); the detailed versioned failure analyses are in `reports/v21/` through `reports/v25/`.

## Data design and temporal model

The repository separates source acquisition, staged lakehouse data, point-in-time features, and synthetic cohorts under `src/controlflow/data/` and `src/controlflow/features/`. Synthetic evaluator truth is deterministic and physically separate from runtime cases. Training, calibration, validation, qualification, and final have distinct roles; the final truth is evaluator-only. The V2.2 independence report documents the development TRAIN/CALIBRATION/VALIDATION separation and runtime exclusion of forbidden truth fields ([`reports/v22/01_data_independence.md`](reports/v22/01_data_independence.md)).

Policy and evidence retrieval is temporal as well as authorization filtered. The candidate sees a bitemporal corpus with stale, future, corrected, current, and irrelevant records, while expected policy IDs come from an evaluator-only declarative fixture ([`reports/v22/08_temporal_oracle.md`](reports/v22/08_temporal_oracle.md)). This tests the policy version available at the case's point in time, rather than rewarding a current policy retrieved after the fact. V26FINAL's stale-policy stratum and temporal accuracy are reported below.

## Modeling, calibration, and selective prediction

The selected V2.2 model bundle combines runtime-observable numeric, categorical, and narrative features. The recorded critical model is a logistic/XGBoost blend and the noncritical model is unweighted logistic; the bundle SHA-256 binds model and configuration files ([`reports/v22/02_model_training.md`](reports/v22/02_model_training.md), [`state/v22_model_bundle.json`](state/v22_model_bundle.json)). Training fits models, validation selects architectures, and calibration selects the operating point. The recorded critical calibrator is isotonic with threshold `0.08333333333333333`, chosen by a predeclared calibration-only rule requiring critical recall at least 0.95 before optimizing false-positive and residual-critical-risk costs ([`reports/v22/03_calibration_threshold.md`](reports/v22/03_calibration_threshold.md)). Risk estimates can route or defer work; they are not action authority.

The local NVIDIA RTX 4090 Laptop GPU evidence is specific to this study. The model report records CUDA XGBoost training, while the runtime bundle binds tabular inference to CPU so the WSL vLLM server owns VRAM ([`reports/v22/02_model_training.md`](reports/v22/02_model_training.md)). No other hardware result is inferred.

## Retrieval, agent, and tool boundaries

The runtime combines risk estimates with point-in-time, scope-filtered evidence and policy retrieval. The bounded agent produces structured explanation data. Evidence verification checks claims and references before an action proposal reaches authorization. SQL tools are read-only with parsed/allowlisted statements, row limits, and timeouts; actions use a typed registry. The LLM does not create truth labels or grant privileges. Relevant implementation is in `src/controlflow/retrieval/`, `src/controlflow/agents/`, `src/controlflow/tools/`, and `src/controlflow/verification/`; the V2.1 security and rationale reports record the design evolution in `reports/v21/`.

## PDP/PEP and approval protocol

The policy enforcement point (PEP) canonicalizes a typed action, uses an explicit registry, and requests an authoritative policy decision point (PDP) decision from authenticated context. The PDP is queried again inside the transaction against current authorization state ([`reports/v22/05_pdp_pep.md`](reports/v22/05_pdp_pep.md)). A separate approval issuer holds a local Ed25519 private key excluded from Git. The bundle and PEP bind only its public key. Approval tokens bind case, action, policy decision, authorization snapshot, versions, reviewer, expiry, and nonce; tampering and replay are negative-tested ([`reports/v22/06_signed_approvals.md`](reports/v22/06_signed_approvals.md)). No real approval or financial transaction is performed.

## Transactional execution and audit integrity

The simulated executor uses SQLite `BEGIN IMMEDIATE` to cover revalidation, approval verification and consumption, state mutation, and hash-chain ledger append in one transaction. Idempotency keys prevent duplicate simulated commits. The chain is tamper-evident only while its head anchor is trusted; it is not a distributed immutable ledger ([`reports/v22/07_transactional_executor.md`](reports/v22/07_transactional_executor.md)).

V2.4 added explicit SQLite connection ownership, deterministic close, WAL checkpoint, integrity and foreign-key checks, and post-close file-stability checks ([`reports/v24/05_sqlite_finalization.md`](reports/v24/05_sqlite_finalization.md)). V26FINAL's recorded finalization, ledger verification, receipt, post-close verifier, and terminal decision all passed ([`reports/v26/release.md`](reports/v26/release.md), [`state/v26_release_decision.json`](state/v26_release_decision.json)). The final release report records 255 review commits matched by 255 approval consumptions; this is a simulation-specific integrity result.

## Evaluation protocol and qualification method

The method freezes candidate code, policies, gates, denominator contracts, and artifact bindings before sealed evaluation. Structural admission verifies holdout composition and independent opportunity counts. Zero-denominator metrics fail rather than appearing as zero error. V2.4 added predeclared minimum denominators for critical, stale-policy, REQUIRE_REVIEW, and DENY opportunities ([`reports/v24/02_denominator_contracts.md`](reports/v24/02_denominator_contracts.md)); V2.6 uses its own frozen gate and stratum files in `configs/v26/`.

Development and failed qualification results remain diagnostic. V2.2 qualification failed serving constraints ([`reports/v22/16_limitations.md`](reports/v22/16_limitations.md)). V2.3 qualification had an unestimable stale-policy gate and unstable SQLite binding ([`reports/v23/14_limitations.md`](reports/v23/14_limitations.md)). V2.4's otherwise completed run was inadmissible after post-close verification failed ([`reports/v24/14_limitations.md`](reports/v24/14_limitations.md)). V2.5's frozen runner failed before candidate access because it compared contamination reports including a volatile timestamp; that qualification identity was permanently consumed and no final holdout was run ([`TECHNICAL_REPORT_V25.md`](TECHNICAL_REPORT_V25.md)). These outcomes drove structural fixes and new holdouts, not reinterpretation of a failed identity.

V2.5 introduced a typed checkpoint schema and a real development closure rehearsal ([`reports/v25/02_checkpoint_contract.md`](reports/v25/02_checkpoint_contract.md)). V2.6 separated stored contamination-report provenance from the independently recomputed semantic digest, then ran a fresh V26QUAL. Independent post-qualification correctness, methodology, and security/reliability reviews cleared before a new frozen final identity was generated. V26FINAL then ran once after deterministic freeze; its manifests, hashes, reviews, and decision are bound in `state/v26_*` and summarized in [`reports/v26/release.md`](reports/v26/release.md).

## Final evaluation

| Frozen V26FINAL measure | Result | Evidence |
| --- | ---: | --- |
| Core STC | 489/600 = 81.50% | `results/v26/final/denominators.json` |
| Binary critical recall | 141/144 = 97.92% | same |
| Typed CRITICAL recall | 141/144 = 97.92% | same |
| Stale-policy errors | 0/150 | same |
| Structured-output failures | 0/600 | same |
| Unauthorized commits | 0/83 | same |
| Approval-bypass commits | 0/255 | same |
| Duplicate commits | 0/600 | same |
| Temporal policy accuracy | 600/600 = 100% | same |
| Evidence completeness | 1005/1005 = 100% | same |
| Total P95, concurrency 2 | 7.392 seconds | `results/v26/final/gate_decision.json` |
| Frozen gates | 17/17 passed | `state/v26_release_decision.json` |

Structural admission was ADMITTED for all 600 cases. The final denominator report is VALID with no aggregate mismatch. The gate decision also records zero leakage findings, artifact-binding violations, checkpoint violations, ledger tamper-verification failures, unresolved BLOCKER findings, and unresolved HIGH findings. These are observations within the sealed synthetic test and the frozen checks, not general safety guarantees.

## Latency, systems results, and supported comparisons

V2.3 recorded request timestamps, token counts, vLLM timings, metrics snapshots, and NVIDIA samples for development-only attribution ([`reports/v23/01_latency_attribution.md`](reports/v23/01_latency_attribution.md)). Its paired serving ablations compared schema size, token cap, batched-token setting, warm versus cold starts, and serving mode. The paired table reports a 14.766-second lower P95 for the minimal-schema candidate than the current-schema candidate on the specified 60-case development comparison, with unchanged Core STC and structured-failure rates in that pair ([`reports/v23/09_ablations.md`](reports/v23/09_ablations.md), [`results/v23/ablations.json`](results/v23/ablations.json)). Those development comparisons motivated a configuration; they are not V26FINAL estimates. The final 7.392-second P95 at observed maximum LLM concurrency 2 is the frozen result, measured on the local laptop GPU configuration.

## Reproducibility and artifact integrity

The promoted research release is the annotated `controlflow-g-v26-promote` tag at `f21a4192373c5bcb815e12a7fa10ac8f183b482b`. The separate `controlflow-g-v26-public` tag captures the later MIT-licensed distribution package; it does not alter the research result. [`state/v26_release_decision.json`](state/v26_release_decision.json) binds the qualification and final identities, freeze hashes, gate decision, denominators, receipts, and review outcomes. The V2.6 manifest and receipt graph binds files by SHA-256 and size; the final SQLite artifact is checked after close. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for portable checks. V26QUAL and V26FINAL are consumed historical evidence and must not be rerun as fresh evidence.

## Cross-platform portability attestation

The additive post-release commit `d524fb7d7a00c2e38920a3f64237a7cba2c1c45a` supports clean-clone CI. Some original Windows frozen files had CRLF or mixed line endings while Git stored normalized LF blobs. [`reports/ci_portability/v26_eol_attestation.json`](reports/ci_portability/v26_eol_attestation.json) records the deterministic reconstruction of original frozen bytes from tagged blobs. [`scripts/verify_release_portability.py`](scripts/verify_release_portability.py) verifies both the tag/blob relationship and the original receipt graph in a disposable tree. It does not rewrite frozen manifests or receipts, and it does not move the research tag.

## Limitations and future work

The data and evaluator truth are synthetic, with public-data inputs; generalization to actual bank controls is untested. The local RTX 4090 Laptop GPU and host conditions limit latency transferability. A zero observed unauthorized or duplicate commit count does not prove zero true failure probability. The local ledger requires a trusted external head anchor for tamper evidence. The repository is not a bank system, does not execute real financial actions, and has no production authorization. [LICENSE](LICENSE) supplies the MIT distribution terms for the public package. Future work could test independent organizations, broader policy catalogs, external anchoring, and hardware/operational variability, but those are proposals rather than measured V2.6 results.
