# ControlFlow-G V2.6: Reliability-First Evaluation of a Governed Agentic Workflow

**Technical Report**  
Synthetic research release; not a bank deployment and not authorized for real financial action.

Repository: https://github.com/ReviveCoding/controlflow-g  
Frozen research tag: `controlflow-g-v26-promote` @ `f21a4192373c5bcb815e12a7fa10ac8f183b482b`  
Public package tag: `controlflow-g-v26-public` @ `00682c88766ec5fdae4b71eda89bfa170e5f5fce`

## Abstract

ControlFlow-G evaluates whether an agentic workflow can remain useful while separating learned risk estimation and language-model reasoning from deterministic temporal evidence controls, authorization, human approval, transactional execution, and independently frozen evaluation. The system is tested in a synthetic financial-services control-exception setting grounded by public regulatory and control corpora. Development experiments screened agent architectures, critical-risk models, calibration choices, and local vLLM serving configurations. Reliability hardening then treated evaluation and artifact-integrity failures as first-class failure modes: leaked truth, invalid denominators, mutable SQLite state, checkpoint-schema mismatch, and semantic-versus-provenance confusion each required structural repair and a fresh holdout rather than reinterpretation of a consumed run. In the frozen one-shot V26FINAL evaluation (600 cases), Core Safe Task Completion was 81.50% (95% interval 78.20%-84.40%), binary and typed critical recall were 97.92% (94.05%-99.29%), stale-policy errors were 0/150, unauthorized commits were 0/83, approval bypasses were 0/255, structured-output failures were 0/600, temporal policy accuracy was 600/600, evidence completeness was 1005/1005, and total P95 latency at observed LLM concurrency two was 7.392 s. All 17 predeclared release gates passed. These results support the feasibility of the reliability-first architecture in the frozen synthetic benchmark; they do not establish production safety, real-bank validity, or zero true risk.

## Executive Summary

The central engineering question was not whether a language model could produce plausible prose. It was whether a useful investigation workflow could preserve correct evidence, point-in-time policy, authorization, review, exactly-once simulated execution, and independently auditable evaluation at the same time. ControlFlow-G therefore places the LLM inside a governed shell: probabilistic components estimate and explain; deterministic components authorize, enforce, commit, and verify.

The report distinguishes five evidence levels: E0 protocol-only comparisons; E1 development diagnostics and ablations; E2 frozen qualification; E3 one-shot final evaluation; and E4 release-assurance evidence. Development results are never presented as final evidence.

| Final V26FINAL measure | Result |

| --- | --- |

| Core STC | 489/600 = 81.50% |

| Binary critical recall | 141/144 = 97.92% |

| Typed CRITICAL recall | 141/144 = 97.92% |

| Stale-policy error rate | 0/150 = 0% |

| Structured-output failure | 0/600 = 0% |

| Temporal policy accuracy | 600/600 = 100% |

| Evidence completeness | 1005/1005 = 100% |

| Unauthorized committed actions | 0/83 |

| Approval-bypass commits | 0/255 |

| Duplicate commits | 0/600 |

| Concurrency-2 total P95 | 7.392 s |



## 1. Introduction

Agentic systems can fail even when their generated text appears correct. In governed workflows, a system may retrieve the wrong policy version, use evidence unavailable at the historical decision time, miss a critical case, trust an adversarial instruction embedded in retrieved content, exceed its authorization, replay an approval, duplicate an action, or produce an evaluation artifact that cannot be independently verified. ControlFlow-G treats those failure modes as part of the system rather than as post-hoc operational concerns.

Safe Task Completion (STC) is intentionally stricter than nominal task success. A case counts as safe only when the task result, evidence, temporal context, authorization, and critical-policy constraints are all correct. This compound definition prevents a fluent but stale, unauthorized, or unsupported answer from being scored as successful.

### 1.1 Research Questions

- RQ1 - Task capability: Can a governed architecture complete useful investigation tasks under bounded evidence and policy context?

- RQ2 - Critical-risk detection: Can critical cases be captured at a predeclared high-recall operating point without delegating action authority to the model?

- RQ3 - Temporal and governance reliability: Can the system select point-in-time policy and enforce authorization, approval, and idempotency constraints?

- RQ4 - Operational efficiency: Can the added governance and verification layers remain feasible on a single local GPU under a latency gate?

- RQ5 - Evaluation integrity: Can the evaluation itself fail closed against contamination, zero denominators, mutable artifacts, schema mismatch, and cross-platform byte drift?



### 1.2 Contributions

- A governed agent architecture that separates learned estimation/reasoning from deterministic authorization and execution authority.

- A temporal benchmark design that physically separates evaluator truth from runtime inputs and tests stale/future/current policy selection.

- Risk-constrained model selection and calibration in which recall constraints precede operating-cost optimization.

- A frozen qualification/final methodology with denominator contracts, consumed holdouts, typed checkpoints, SQLite finalization, artifact receipts, and independent review.

- A failure-driven hardening record showing that evaluation-integrity defects can be as consequential as model-quality defects.

- A public reproducibility package that preserves immutable research evidence while providing additive cross-platform portability verification.



## 2. Evidence Hierarchy and Reporting Convention

| Level | Meaning | Examples |

| --- | --- | --- |

| E0 | Protocol only | Planned baseline with no executed quantitative evidence |

| E1 | Development evidence | Architecture screens, model tournaments, serving ablations |

| E2 | Qualification evidence | Fresh frozen V26QUAL |

| E3 | Final evidence | One-shot V26FINAL |

| E4 | Release assurance | Independent reviews, artifact bindings, receipts, portability |



This hierarchy is essential to the interpretation of the report. It prevents a development optimization from being mistaken for a final benchmark result and prevents a protocol-defined baseline from receiving invented numerical performance.

## 3. Related Evaluation Frameworks and Reporting Principles

The reporting structure is informed by, but does not claim certification or compliance with, external evaluation guidance. NIST AI RMF 1.0 emphasizes rigorous measurement, uncertainty, benchmark comparison, documentation, and independent review. NIST TEVV-Athlon frames AI assessment as a customizable measurement process suitable for systems including agentic AI. NIST ARIA combines model testing, red teaming, and user testing; ControlFlow-G covers extensive automated system testing and adversarial/failure testing, while real user testing remains a limitation and priority extension. The NeurIPS checklist motivates explicit experimental detail, uncertainty, compute disclosure, and reproducibility. Model Cards motivate clear intended-use and limitation statements.

## 4. Problem Setting and Threat Model

The target workflow investigates synthetic control and operational exceptions. An exception record and accessible evidence must be transformed into risk classification, evidence-grounded explanation, policy-grounded disposition, and a simulated state transition. The design assumes retrieved text is untrusted data, not executable instruction, and assumes high-risk actions require independently enforced policy and, where applicable, human approval.

| Threat class | Representative failure | Primary control |

| --- | --- | --- |

| Truth leakage | Evaluator-only target exposed at runtime | Physical runtime/truth separation |

| Critical miss | CRITICAL case routed as routine | High-recall calibrated critical model |

| Temporal error | Current or future policy used for historical event | Bitemporal retrieval and temporal oracle |

| Evidence failure | Missing/conflicting/adversarial evidence | Evidence verification and insufficiency state |

| Prompt injection | Retrieved text instructs the agent to override policy | Retrieved text treated as evidence only; verifier + PEP |

| Tool misuse | Unbounded SQL or action request | Parsed read-only tools and typed action registry |

| Authorization failure | Action exceeds identity/scope/purpose | Authoritative PDP + PEP revalidation |

| Approval failure | Replay, payload substitution, bypass | Signed approval token + transactional consumption |

| Execution failure | Duplicate side effect | Idempotency + SQLite transaction |

| Audit failure | Ledger mutation or unstable artifact | Hash-chain verification + post-close stability |

| Evaluation failure | Zero denominator / contamination / schema drift | Denominator contracts + typed checkpoint + frozen manifests |



## 5. Data and Benchmark Construction

The benchmark combines public-source grounding with deterministic synthetic enterprise truth. Public corpora are used as controls, policy, long-document context, and descriptive data; they are not treated as enterprise ground truth. Synthetic evaluator truth is physically separated from runtime cases.

| Source | Recorded scale / role |

| --- | --- |

| CFPB Consumer Complaint Database | 17,651,452 records through 2026-09-09; 3,849,642 historical narratives (21.81%); descriptive public corpus, not population prevalence |

| NIST OSCAL / SP 800-53 Rev. 5.1.1 | 1,193 base controls + enhancements; 3,418 relationship links |

| NIST SP 800-53A | 5,383 assessment rows |

| Title 12 CFR / eCFR + GovInfo | Three dated eCFR snapshots + ten official 2024 volumes; 43,768 section-like elements |

| SEC EDGAR bounded corpus | 12 10-K/10-Q primary filings across three financial-sector issuers; ~119 MB |

| Synthetic benchmark generator | Deterministic case truth, evidence, temporal scenarios, authorization opportunities, adversarial variants |



### 5.1 Split Roles

| Split | Role | May influence selection? |

| --- | --- | --- |

| TRAIN | Fit model parameters | Yes |

| CALIBRATION | Fit calibration and choose operating threshold | Yes, calibration only |

| VALIDATION | Choose architectures/configurations | Yes |

| DEVELOPMENT | Diagnostics, security tests, ablations | Yes, development only |

| V26QUAL | Frozen eligibility evidence | No tuning after observation |

| V26FINAL | One-shot final confirmation | No tuning; consumed after use |



## 6. System Architecture

ControlFlow-G uses a deterministic shell around probabilistic components. Risk models estimate severity and criticality. Temporal and authorization filters constrain retrieval. The LLM produces a bounded structured explanation but does not define evaluator truth, grant privileges, or authorize a commit. Evidence verification precedes action authorization. PDP and PEP decisions are rechecked inside the transaction. Approved simulated changes are idempotent and appended to a tamper-evident audit chain.

## 7. Risk Modeling, Calibration, and Selective Routing

Development model selection was not a simple leaderboard exercise. The project compared tabular, text, neural, XGBoost, and fusion candidates, then froze a later bundle that balanced critical recall, calibration, review burden, robustness, and operational constraints. V2.2 ultimately bound a logistic/XGBoost critical blend and an unweighted logistic noncritical route. Isotonic calibration and a 0.08333 critical threshold were selected using CALIBRATION only; the predeclared rule first required critical recall >= 0.95, then minimized false-positive and residual-critical-risk costs with an additional penalty below a development safety margin.

### 7.1 Development Critical-Risk Comparison (E1)

| Model | Critical recall | AUPRC | Brier | ECE | Review rate |

| --- | --- | --- | --- | --- | --- |

| Learned fusion | 1.0000 | 0.9612 | 0.0222 | 0.0136 | 0.2653 |

| Late probability fusion | 0.9847 | 0.8549 | 0.0610 | 0.0595 | 0.3400 |

| Focal-loss PyTorch MLP (CUDA) | 0.9847 | 0.7646 | 0.0840 | 0.0476 | 0.4312 |

| Weighted logistic | 0.9847 | 0.7946 | 0.0799 | 0.0284 | 0.4461 |

| Text linear | 0.9771 | 0.6285 | 0.1129 | 0.0462 | 0.8607 |

| Weighted XGBoost (CUDA) | 1.0000 | 0.8081 | 0.0770 | 0.0336 | 1.0000 |



These values are development-only evidence. They illustrate the performance/review/calibration trade space; they are not V26FINAL estimates.

## 8. Temporal Retrieval, Evidence Verification, and Governance

The candidate operates over a bitemporal corpus containing current, stale, future, correction, and irrelevant records. Expected policy IDs are held in evaluator-only fixtures. The runtime filters on time and authorization before retrieval, and evidence references are verified before an action proposal can proceed. The resulting authorization state is explicit: AUTO, REVIEW_REQUIRED, INSUFFICIENT_EVIDENCE, or DENY.

### 8.1 PDP, PEP, Approval, and Transactional Execution

The PEP canonicalizes typed actions and obtains an authoritative PDP decision from authenticated context. Sensitive actions require signed approval. The approval token binds case, normalized action, policy decision, authorization snapshot, reviewer, expiry, nonce, and versions. During simulated execution, SQLite BEGIN IMMEDIATE covers policy revalidation, approval verification/consumption, state mutation, idempotency, and ledger append as one transaction. The audit chain is tamper-evident only while its head anchor remains trusted; it is not a distributed immutable ledger.

## 9. Evaluation Design and Measurement Methodology

The evaluation protocol freezes candidate code, model bundles, thresholds, retrieval, policy, prompt/schema, LLM revision, denominator contracts, gates, and seeds before qualification. Structural admission validates holdout composition before candidate access. Required rates use explicit minimum denominators and null-policy FAIL: at least 120 critical cases, 150 stale-policy opportunities, 100 authoritative REVIEW_REQUIRED opportunities, and 50 DENY opportunities. A consumed holdout is never repaired and rerun as fresh evidence.

### 9.1 Safe Task Completion

STC is a compound system metric rather than an answer-accuracy metric: TaskCorrect AND EvidenceCorrect AND TemporalCorrect AND Authorized AND NoCriticalPolicyViolation. A case with a plausible answer but stale evidence, unauthorized execution, or a critical policy violation does not count as safely completed.

### 9.2 Independent Review and Artifact Closure

Correctness, methodology, and security/reliability reviewers must clear BLOCKER and HIGH findings before protected transitions. Checkpoints are typed and validated. SQLite remains mutable until all application connections close, integrity checks pass, WAL checkpoint(TRUNCATE) completes, and a separate hash-only stability verifier observes stable bytes. Artifact graphs bind substantive evidence by size and SHA-256, followed by post-close verification, closure receipts, and a terminal verifier.

## 10. Development Experiments

### 10.1 Architecture Screening (E1)

| Architecture | Cases | Structured failure | Critical recall | STC | P95 latency |

| --- | --- | --- | --- | --- | --- |

| 0.5B unconstrained | 32 | 100% | 0% | 0% | 1.855 s |

| Qwen3 unconstrained | 32 | 100% | 0% | 0% | 11.976 s |

| Qwen3 + schema | 32 | 0% | 71.43% | 25.00% | 15.255 s |

| Decomposed flat | 32 | 0% | 85.71% | 84.38% | 17.944 s |

| Decomposed hierarchical | 32 | 0% | 100% | 84.38% | 17.358 s |



The small 32-case screen is architecture selection evidence only. Its main value is directional: unconstrained generation failed the structured contract, schema constraints removed formatting failure, and decomposed deterministic structure materially improved the compound task metric in development.

### 10.2 Serving and Output-Contract Ablations (E1)

The V2.3 serving investigation showed that the output contract dominated tail latency. In a paired 60-case development comparison, replacing the current schema with the minimal schema reduced P95 from 23.700 s to 8.933 s (62.3%) with no Core STC or structured-failure degradation in that pair. Reducing the token cap from 160 to 128 further reduced P95 from 8.933 s to 5.826 s (34.8%). A 2,048 batched-token budget outperformed both 1,024 (6.825 s) and 4,096 (7.044 s) in the compared configurations, showing that larger batching was not monotonically better.

Across different frozen cohorts, V22QUAL observed P95 23.024 s and failed the <=15 s latency gate, whereas V26FINAL observed 7.392 s. Because these are different holdouts and later software states, this cross-version change is descriptive rather than a causal paired estimate.

## 11. Reliability Hardening Through Consumed Failures

ControlFlow-G treats failures in the evaluation pipeline as genuine research outcomes. Several of the most consequential defects were not model-score failures; they were failures of validity, provenance, or closure. Each affected qualification identity remained consumed. The repair pattern was defect-class identification -> structural fix -> new review/freeze -> fresh holdout.

| Version | Failure class | Why it mattered | Structural response |

| --- | --- | --- | --- |

| V2.1 | Truth and approval boundaries | Risk of leakage or self-authorized action | Physical truth separation; authorization redesign |

| V2.2 | Sustained latency + structured failures | Release gates can fail despite strong task metrics | Latency attribution and structured-output redesign |

| V2.3 | Zero stale-policy denominator | A 0/0 error rate is not evidence of zero error | Predeclared denominator contracts; null-policy FAIL |

| V2.3 | SQLite lifecycle instability | Byte binding occurred before the DB reached a stable post-close state | Explicit connection ownership, finalizer, WAL truncation, hash stability |

| V2.4 | Checkpoint schema mismatch | Frozen verifier parsed a producer artifact incorrectly | Typed checkpoint contract shared by producer/consumer |

| V2.5 | Volatile provenance equality | Timestamp changes falsely invalidated semantically identical evidence | Separate stable semantic digest from volatile provenance |

| V2.6 | Closure | All repaired paths required fresh evidence | Fresh V26QUAL, review, fresh one-shot V26FINAL |



## 12. Frozen Qualification and Final Results

V26QUAL (seed 26061) and V26FINAL (seed 26062) each contained 600 cases and completed the sealed path. Qualification was used only to establish eligibility; after it passed, independent post-qualification reviews cleared before the final identity was generated and frozen. Final execution then ran once.

| Metric | V26QUAL (E2) | V26FINAL (E3) | Change |

| --- | --- | --- | --- |

| Core STC | 491/600 = 81.83% | 489/600 = 81.50% | -0.33 pp |

| Binary critical recall | 134/136 = 98.53% | 141/144 = 97.92% | -0.61 pp |

| Typed critical recall | 134/136 = 98.53% | 141/144 = 97.92% | -0.61 pp |

| Stale-policy errors | 0/150 | 0/150 | 0 |

| Structured-output failures | 0/600 | 0/600 | 0 |

| Unauthorized commits | 0/81 | 0/83 | 0 |

| Approval-bypass commits | 0/255 | 0/255 | 0 |

| P95 latency @ concurrency 2 | 6.833 s | 7.392 s | +0.559 s |



The final did not show a large collapse relative to qualification: STC changed by -0.33 percentage points and critical recall by -0.61 percentage points, while P95 latency increased by 0.559 s. This is best interpreted as holdout consistency within the same synthetic benchmark design, not as independent real-world replication.

### 12.1 Final 17 Release Gates

| Gate | Observed | Threshold | Decision |

| --- | --- | --- | --- |

| Core STC | 489/600 = 81.50% | >= 80% | PASS |

| Binary critical recall | 141/144 = 97.92% | >= 95% | PASS |

| Typed CRITICAL recall | 141/144 = 97.92% | >= 92% | PASS |

| Stale-policy error rate | 0/150 = 0% | <= 2% | PASS |

| Structured-output failure | 0/600 = 0% | <= 1% | PASS |

| Temporal policy accuracy | 600/600 = 100% | >= 98% | PASS |

| Evidence completeness | 1005/1005 = 100% | >= 98% | PASS |

| Unauthorized committed actions | 0/83 | = 0 | PASS |

| Approval-bypass commits | 0/255 | = 0 | PASS |

| Duplicate commits | 0/600 | = 0 | PASS |

| Concurrency-2 total P95 | 7.392 s | <= 15 s | PASS |

| Leakage findings | 0 | = 0 | PASS |

| Checkpoint violations | 0 | = 0 | PASS |

| Artifact-binding violations | 0 | = 0 | PASS |

| Ledger tamper-verification failures | 0 | = 0 | PASS |

| Unresolved BLOCKER findings | 0 | = 0 | PASS |

| Unresolved HIGH findings | 0 | = 0 | PASS |



## 13. Statistical Interpretation and Uncertainty

Point estimates alone overstate certainty, particularly when the observed failure count is zero. The frozen denominator artifact reports interval estimates for the principal ratio metrics. For zero-event outcomes, the upper confidence bound is more informative than the 0% point estimate.

| Metric | Point estimate | 95% interval / upper bound | Denominator |

| --- | --- | --- | --- |

| Core STC | 81.50% | 78.20% - 84.40% | 600 |

| Binary critical recall | 97.92% | 94.05% - 99.29% | 144 |

| Typed critical recall | 97.92% | 94.05% - 99.29% | 144 |

| Evidence completeness | 100% | 99.62% - 100% | 1005 |

| Temporal accuracy | 100% | 99.36% - 100% | 600 |

| Stale-policy error | 0% | upper 95% ~ 2.50% | 150 |

| Unauthorized commit rate | 0% | upper 95% ~ 4.42% | 83 |

| Approval-bypass rate | 0% | upper 95% ~ 1.48% | 255 |

| Duplicate-commit rate | 0% | upper 95% ~ 0.64% | 600 |



The critical recall interval is materially wider than the STC interval because only 144 final cases are critical. The zero unauthorized-commit observation (0/83) is consistent with true rates above zero; the frozen interval has an upper bound of approximately 4.42%. The correct claim is therefore “zero observed unauthorized commits in 83 certified opportunities,” not “zero risk.”

## 14. Operational Interpretation

| Technical metric | Operational interpretation |

| --- | --- |

| Critical recall | Residual risk of missing high-severity exceptions |

| REVIEW_REQUIRED / approval flow | Human review workload and escalation burden |

| DENY / unauthorized commit | Effectiveness of action-boundary enforcement |

| Temporal accuracy / stale-policy error | Correctness of historical policy reconstruction |

| Evidence completeness | Auditability and support for reviewer reconstruction |

| P95 latency | Interactive feasibility for analyst-facing investigation |

| Duplicate commit rate | Exactly-once behavior at the business-action layer |



The frozen study did not measure real analyst time savings, adoption, trust calibration, or organizational economics. Those quantities should not be inferred from the technical metrics.

## 15. Reproducibility, Compute, and Artifact Integrity

| Item | Frozen / public value |

| --- | --- |

| Research tag | controlflow-g-v26-promote @ f21a4192373c5bcb815e12a7fa10ac8f183b482b |

| Public package tag | controlflow-g-v26-public @ 00682c88766ec5fdae4b71eda89bfa170e5f5fce |

| Python | 3.11.9 in final environment manifest |

| LLM | Qwen/Qwen3-4B-Instruct-2507 |

| LLM revision | cdbee75f17c01a7cc42f958dc650907174af0554 |

| vLLM | 0.29.0 |

| Precision / backend | BF16 / xgrammar |

| Max model length | 4096 |

| Max sequences | 2 |

| Max batched tokens | 2048 |

| Output token cap | 128 |

| Serving mode | interactivity; optimization level 2 |

| Observed LLM concurrency | 2 |

| GPU | NVIDIA GeForce RTX 4090 Laptop GPU, ~16 GB VRAM |



The public package preserves the frozen research tag and adds a deterministic line-ending portability attestation. Some historical Windows artifacts were bound with CRLF bytes while Git stored normalized LF blobs; the portability verifier reconstructs original bytes in a disposable tree and verifies the original receipt graphs without rewriting the research evidence.

## 16. Limitations

- Synthetic enterprise truth: benchmark validity does not establish performance on proprietary bank cases, policies, incentives, or organizational practices.

- No real human user study: simulated REVIEW_REQUIRED behavior does not establish analyst review time, override quality, cognitive load, or trust calibration.

- Single local hardware context: latency and throughput were measured on one RTX 4090 Laptop GPU at observed concurrency two; cluster-scale performance is unknown.

- One pinned local LLM configuration: model-family robustness and cross-model generalization are not established.

- Local governance infrastructure: Ed25519 approvals, SQLite execution, and a local audit chain demonstrate contracts but are not substitutes for enterprise IAM, HSM/KMS, WORM logging, or distributed transaction guarantees.

- Trusted-head assumption: the hash chain is tamper-evident only while its external head anchor remains trusted.

- Zero observed critical security failures do not prove zero true failure probability; confidence bounds remain non-zero.

- Public-source corpora have their own coverage and representativeness limits; CFPB complaints are not population prevalence and narratives changed publication policy in 2026.



## 17. Future Extensions

| Extension axis | Priority experiments |

| --- | --- |

| Human factors | Compliance-analyst user study: review time, override rate, inter-rater agreement, trust calibration, workload |

| Data / domain | De-identified real exception cases; broader policy catalogs; multi-institution evaluation; jurisdictional temporal histories |

| Modeling | Cross-model comparison; smaller/larger open LLMs; selective prediction; conformal or uncertainty-aware routing; carefully controlled fine-tuning |

| Governance | Enterprise IAM, policy-as-code engines, HSM/KMS-backed approvals, external audit-head anchoring, WORM evidence stores |

| Systems | Higher concurrency, multi-GPU serving, SLO-aware scheduling, distributed recovery, process/node fault injection |

| Evaluation science | Independent third-party TEVV, adaptive red teaming, longitudinal drift, richer adversarial challenges, real user testing |



The highest-value missing layer is real user testing. NIST ARIA-style evaluation explicitly combines model testing, red teaming, and user testing; ControlFlow-G currently has strong automated system testing and adversarial/failure testing but lacks empirical human-use evidence.

## 18. Conclusion

ControlFlow-G demonstrates that a useful agentic workflow can be evaluated as a governed system rather than as a language model in isolation. In the frozen synthetic V26FINAL benchmark, the architecture achieved 81.50% Core STC, 97.92% critical recall, perfect observed temporal accuracy and evidence completeness, zero observed unauthorized commits, approval bypasses, duplicate commits, stale-policy errors, and structured-output failures in their respective final opportunity sets, while remaining below an operational P95 gate at 7.392 s on a local single-GPU setup. All 17 frozen release gates passed.

The more important result is methodological: several high-impact failures were discovered in evaluation validity and artifact integrity rather than in model quality. Treating those failures as first-class evidence, consuming affected holdouts, and requiring fresh qualification/final identities produced a stronger and more auditable release process. The resulting evidence supports a reproducible synthetic research release; it does not establish production fitness or zero true risk.

## References

[1] NIST, Artificial Intelligence Risk Management Framework (AI RMF 1.0), 2023. https://doi.org/10.6028/NIST.AI.100-1

[2] NIST, The TEVV-Athlon Framework for Evaluating AI Systems, initial public draft, 2026. https://www.nist.gov/artificial-intelligence/ai-research/tevv-athlon-framework-evaluating-ai-systems

[3] Jensen et al., ARIA Evaluation Planning Manual: Elements of ARIA-Style AI Evaluations, NIST AI 200-3, 2026. https://doi.org/10.6028/NIST.AI.200-3

[4] Amironesei et al., Assessing Risks and Impacts of AI (ARIA): Pilot Evaluation Report, NIST AI 700-2, 2025. https://doi.org/10.6028/NIST.AI.700-2

[5] NeurIPS, Paper Checklist Guidelines. https://neurips.cc/public/guides/PaperChecklist

[6] Mitchell et al., Model Cards for Model Reporting, 2019. https://research.google/pubs/model-cards-for-model-reporting/

[7] Keller et al., Expanding the AI Evaluation Toolbox with Statistical Models, NIST AI 800-3, 2026. https://www.nist.gov/publications/expanding-ai-evaluation-toolbox-statistical-models



## Appendix A. Repository Evidence Map

| Evidence topic | Primary repository artifact |

| --- | --- |

| Research release decision | state/v26_release_decision.json |

| Final gates | results/v26/final/gate_decision.json |

| Final denominators / intervals | results/v26/final/denominators.json |

| Qualification gates | results/v26/qualification/gate_decision.json |

| Qualification denominators | results/v26/qualification/denominators.json |

| Final environment | state/v26_final_environment_manifest.json |

| Architecture screen | reports/v2/01_model_tournament.md |

| Critical classifier study | reports/v2/02_critical_classifier.md |

| Fusion study | reports/v2/04_fusion.md |

| Serving ablations | reports/v23/09_ablations.md |

| V22 qualification failure | reports/v22/13_internal_qualification.md |

| Denominator contracts | reports/v24/02_denominator_contracts.md |

| Release narrative | reports/v26/release.md |

| Public reproducibility | REPRODUCIBILITY.md |



## Appendix B. Evidence-Level Rule

When interpreting any result, first identify its evidence level. E1 development screens can motivate a design choice but cannot substantiate the final release claim. E2 qualification can establish eligibility but is not the one-shot final. E3 V26FINAL supports the final metric claims. E4 assurance evidence supports the integrity of the release process, not the underlying task metric itself.