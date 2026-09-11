# ControlFlow-G Execution Specification

ControlFlow-G is a production-oriented, public-data research prototype for governed investigation and simulated remediation of enterprise control and operational exceptions.

## Research question

Evaluate—without assuming a positive result—whether point-in-time data, calibrated ML risk, anomaly signals, evidence-grounded retrieval, bounded typed tools, deterministic external authorization, selective human review, durable execution, and auditability improve Safe Task Completion, critical-error risk, review efficiency, latency, cost, and reliability versus less constrained LLM/agent baselines.

Safe Task Completion requires a correct task outcome, correct evidence, correct temporal context, correct authorization, and no critical policy violation. Nominal success and STC are reported separately.

## Required workflow

Exception → identity/purpose → triage → control/regulation discovery → historical retrieval → structured investigation → calibrated risk → anomaly signal → bounded LLM analysis → evidence verification → runtime authorization → AUTO / REVIEW_REQUIRED / INSUFFICIENT_EVIDENCE / DENY → optional review → idempotent simulated action → immutable audit record.

## Data and platform scope

Acquire versioned official subsets of NIST SP 800-53 Rev. 5.1, SP 800-53A, OSCAL controls; CFPB complaints; Title 12 CFR/eCFR current and historical material; and a bounded financial-sector SEC EDGAR corpus. AMLSim is optional; a deterministic internal generator is the fallback. Record source, retrieval time, snapshot, terms reference, schema, files, rows, bytes, hashes, method, and local path. Raw successful downloads are immutable.

Implement strict/evolving/quarantine contracts; RAW→BRONZE→SILVER→GOLD PySpark/Spark SQL pipelines; SCD2 and bitemporal business/system time; CDC/stream replay and recovery; point-in-time features plus an intentionally leaky comparison; deterministic synthetic ground truth; exact split IDs/hashes including a sealed final holdout; and an idempotent action ledger.

## Evaluation scope

- Supervised: rules, logistic regression, random forest, XGBoost CUDA, LightGBM when genuine CUDA support exists, PyTorch MLP, text neural model, and text+structured fusion; shared splits; 5 traditional seeds and at least 3 deep seeds when resources permit.
- Calibration/selectivity: uncalibrated, Platt, isotonic, temperature where appropriate; validation selection only; risk-coverage and HITL reviewer-error sensitivity.
- Anomaly/semi-supervised: thresholds, isolation forest, LOF, CUDA autoencoder; labeled fractions 1/5/10/25/50/100% with paired subsets.
- Retrieval: Boolean, BM25, dense CUDA embeddings, hybrid, reranker, metadata, temporal, and authorization filters; chunking and feasible scale studies.
- Agents: rules/templates, single LLM, RAG, unrestricted ReAct-style, planner/executor, verifier, and full ControlFlow-G using a pinned common base model where possible.
- Security S01-S15; data and system failure suites; drift; Pandas/optional Polars/PySpark/Spark SQL/incremental scalability; AB01-AB12 and specified interactions.
- Use paired bootstrap/permutation/McNemar where suitable, 95% intervals, effect sizes, and Holm correction. Promotion does not depend on p-values alone.

Primary metrics include macro F1, AUROC/AUPRC, critical recall/FNR, Brier/ECE, retrieval ranking and evidence metrics, authorization/tool correctness, STC, critical hallucination and bypass rates, latency percentiles, recovery, tokens/GPU-seconds/cost proxies, and operational utility sensitivity under multiple weights.

## Execution phases

P00 environment/resources/dependencies; P01 desktop study/sources; P02 acquisition/provenance; P03 contracts; P04 EDA; P05 lakehouse; P06 CDC/temporal; P07 point-in-time/leakage; P08 benchmark; P09 splits/sealing; P10 supervised; P11 fusion; P12 calibration/selectivity; P13 anomaly/semi-supervised; P14 baseline retrieval; P15 governed retrieval; P16 agent baselines; P17 full system; P18 authorization/HITL; P19 security; P20 recovery; P21 scaling/cost; P22 ablations; P23 validation statistics; P24 independent reviews/repairs; P25 freeze; P26 one final evaluation; P27 deployment simulation; P28 synthesis; P29 release decision; P30 documentation/cards/resume evidence.

## Freeze and release

P25 freezes source state, dependencies, data and split hashes, features, models/hyperparameters, calibration/thresholds, embedding/reranker, retrieval, prompts/LLM revision, graph, tools, authorization/HITL policies, scorers, gates, and seeds into `state/freeze_manifest.json` with a deterministic hash. P26 then sets `sealed_test_consumed=true`.

Pre-specified gates cover unauthorized irreversible simulated action and approval bypass (zero tolerance), critical recall, stale-policy error, STC, structured-output validity, latency, cost, and regression tolerance. Final decision is exactly PROMOTE, CONDITIONAL_PROMOTE, or NO_PROMOTE.

## Required deliverables

Maintain state/manifests; Parquet result tables; MLflow-compatible experiment records; source, desktop, EDA, engineering, modeling, retrieval, architecture, governance, reliability, ablation, statistics, scaling, final, release, limitation, and future-work reports; README, technical report, model/agent/data cards, risk register, threat model, reproducibility guide; honest figures; and post-final resume evidence linked to exact experiment artifacts.

The complete detailed requirements are the originating execution request; this file preserves its governing constraints and phase contract for resumed sessions.

