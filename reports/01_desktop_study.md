# Control Exception Investigation Desktop Study

## Executive assessment

Large regulated financial institutions investigate control and operational exceptions through a mixture of case systems, policy repositories, data warehouses, analyst judgment, approval queues, and manually assembled evidence. The central difficulty is not simply generating fluent analysis. It is reconstructing what was true and knowable at a historical decision time, limiting access and action to an authenticated purpose, and leaving a defensible record that survives retries, policy changes, and audit.

ControlFlow-G studies a bounded architecture for that problem. It is a public-data, synthetic-ground-truth prototype and production-like simulation. It is not a deployment in a financial institution, does not have private bank data, and cannot execute a real financial action.

## Target workflow and current-state analogue

The target workflow begins when a monitoring rule, control test, complaint pattern, transaction signal, or reviewer creates an exception. Identity and stated purpose establish a runtime boundary. The investigator determines the relevant control and policy version, retrieves related cases, queries governed structured data, estimates severity and anomaly, assembles claims with evidence, and proposes a disposition. Deterministic policy then decides whether the proposal may run automatically, needs review, lacks evidence, or is denied. Approved changes use an idempotent simulated API and produce an immutable audit record.

A plausible manual analogue distributes those steps across ticketing, shared documents, regulatory research, SQL requests, email/approval systems, and several evidence repositories. Analysts repeatedly search for the same material and manually reconcile timestamps. Senior reviewers spend time on low-risk cases because uncertainty is not consistently quantified. Evidence copied into cases loses source/version context, while retries and handoffs create duplicate-action and incomplete-audit risks.

## Stakeholders and responsibilities

| Stakeholder | Primary responsibility | Required assurance |
|---|---|---|
| Control Analyst | Investigate, assemble evidence, propose disposition | Relevant point-in-time evidence and usable explanations |
| Business Owner | Confirm operational facts and own remediation | Clear state transition, impact, and rollback |
| Risk Manager | Set risk appetite and escalation thresholds | Calibrated uncertainty and critical-risk capture |
| Compliance Reviewer | Interpret obligations and approve sensitive actions | Correct jurisdiction/version and complete citations |
| Data Scientist | Develop and evaluate models | Leakage controls, frozen splits, calibration, drift evidence |
| ML Engineer | Package and serve models | Reproducibility, device proof, monitoring, rollback |
| Platform Engineer | Operate data/agent infrastructure | Recovery, capacity, access boundaries, observability |
| Auditor | Reconstruct decisions independently | Tamper-evident lineage, authorization, evidence and action logs |

Humans remain responsible for high-risk approval, policy interpretation where evidence conflicts, exceptions outside the validated operating domain, and any decision with irreversible real-world consequences. Simulated-review experiments do not establish real reviewer performance.

## Data flows and trust boundaries

Public NIST, CFPB, eCFR/GovInfo, and SEC material enters an immutable raw zone through source-specific downloaders. Checksums, retrieval times, source versions, schemas, and terms references travel with each object. Parsed Bronze records retain source fidelity; Silver applies contracts, quarantine, deduplication, semantic normalization, and bitemporal intervals; Gold exposes case, control, regulation, role, prediction, agent, tool, and review entities plus case-360 views.

The highest-risk boundary lies between untrusted narrative/document content and executable behavior. Retrieved text is evidence, never instruction. The LLM never receives unrestricted database credentials and cannot authorize itself. SQL is parsed and read-only, tools are typed and scoped, and proposed writes cross a separate policy/HITL/action boundary. Approval binds the exact normalized action payload and version; any edit requires reevaluation.

## Bottlenecks and AI opportunities

Likely bottlenecks include fragmented search, historical-policy reconstruction, repetitive case triage, inconsistent severity judgments, manual structured-data requests, long evidence assembly, review queues, and failure-prone handoffs. Retrieval and structured features can reduce search effort; calibrated classification and anomaly scores can prioritize work; deterministic templates can handle simple cases; an LLM can synthesize bounded evidence and propose hypotheses; selective prediction can reserve uncertain or high-risk work for people.

Those opportunities create failure modes: confident but unsupported claims, stale-policy citation, hidden future leakage, unauthorized retrieval, prompt injection, unsafe tool arguments, approval bypass, duplicate side effects, and automation bias. Architecture components therefore earn their place only if controlled experiments show that they improve the chosen outcomes at acceptable latency, cost, and review burden.

## Non-functional and audit requirements

- Historical reconstruction must use both business-valid and system-known time.
- Availability and recovery must tolerate duplicate, late, out-of-order, and replayed events plus component timeouts and process crashes.
- Authorization must fail closed and bind identity, role, unit, region, clearance, purpose, session, data class, tool/action risk, severity, and arguments.
- Every output must satisfy a versioned schema; every material claim must map to authorized, temporally valid evidence.
- Simulated side effects must be exactly-once at the business-action layer and offer rollback metadata where declared.
- Raw inputs are immutable; derived artifacts are content-hashed; experiments record data/split/config/model/prompt/policy versions and hardware.
- Operations must respect a 20 GiB C-drive reserve, one-heavy-GPU-job semaphore, resource monitoring, bounded timeouts/retries, and graceful profile reduction.
- Audit logs must be append-oriented, provenance-linked, reconstructable, and free of credentials or unnecessarily exposed restricted text.

## Architecture alternatives

1. **Rules and templates only.** Strong determinism and auditability, low cost, but brittle recall and limited synthesis on novel cases. This is AG0/M0, not a strawman.
2. **Single LLM.** Simple integration and flexible prose, but weak temporal/data grounding and no independent control boundary. It provides an architecture baseline, not an acceptable privileged deployment pattern.
3. **LLM with RAG.** Improves access to source text but does not by itself solve stale evidence, authorization, injection, calibration, or side-effect idempotency.
4. **Unrestricted tool agent.** Potentially flexible, with a deliberately enlarged attack and reliability surface. It is evaluated as AG3, never connected to real actions.
5. **Planner/executor with verifier.** Separates planning and execution and adds evidence checks, but still needs deterministic authorization, HITL, durable state, and action semantics.
6. **Governed deterministic shell with bounded model components.** ControlFlow-G routes predictable steps deterministically and uses ML/LLM components only where probabilistic inference is useful. The extra controls can reduce unsafe completions but may increase latency, review, false blocks, and operational complexity; the study permits a null or negative conclusion.

## Build-versus-buy considerations

Managed lakehouse, catalog, vector search, model serving, policy engines, case management, and observability platforms can reduce undifferentiated operations. They introduce licensing cost, vendor-specific semantics, residency questions, and migration constraints. This study keeps adapters for retrieval, features, tracking, policy, warehouse, and endpoints so the core reproduction needs no paid cloud. Local Delta-compatible/Spark, MLflow, and policy components demonstrate contracts, not parity with every managed-service guarantee. Databricks behavior will be reported only if authenticated execution actually occurs; none was available at preflight.

## Success criteria and decision frame

The principal outcome is Safe Task Completion, not raw answer accuracy. A success must also use correct evidence and time, make the correct authorization decision, and avoid a critical policy violation. The study separately measures nominal success, critical recall/FNR, evidence/citation quality, tool and argument correctness, bypass/exfiltration rates, review efficiency, latency, recovery, GPU time, throughput, and cost proxies. Multiple utility weights explore value disagreements rather than presenting a single arbitrary business score.

Promotion gates are frozen before the final test. Unauthorized irreversible simulated actions and approval bypass have zero tolerance. A strong model score cannot compensate for those failures, while a perfectly safe but unusably slow/high-review system may also fail. `NO_PROMOTE` is a legitimate result.

## Limitations known before experimentation

- Synthetic enterprise labels simplify real institutional policy, incentive, ownership, and data-quality dynamics.
- Public complaints are not population-representative; current CFPB data no longer publish narratives, limiting public-language realism.
- NIST controls and Title 12 passages support grounding tests but do not define the complete control framework of any institution.
- SEC filings are disclosure documents and distractors/context, not internal evidence.
- Simulated users and reviewers cannot establish real human factors, organizational adoption, or production economics.
- A single laptop GPU and local Spark constrain scale and concurrency; results cannot be generalized to a cluster without measurement.
- Local action/audit simulation does not confer the transactional, identity, retention, or legal assurances of a production platform.
