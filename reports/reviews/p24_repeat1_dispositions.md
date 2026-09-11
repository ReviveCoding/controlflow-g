# P24 Repeat-1 Review Dispositions

This records the main agent's independent disposition of the three repeat-review reports against commit
`01b2a870108b16deaedc9f12a7614ac1b364492b`. That snapshot failed P24. No final-test identifiers or outcomes were
read. All findings below were classified before the second stable-snapshot review.

## Correctness review

| Finding | Classification | Resolution |
|---|---|---|
| Freeze inventory omitted final-evaluation inputs and used a time-dependent identity hash | VALID | Freeze identity excludes creation time, hashes the clean Git revision, configs, source, staged NIST, benchmark/splits, calibrated model, agent/security inputs, and rejects changed files. Missing or non-finite release metrics now fail closed. |
| Workflow bypassed typed tools and the final-output schema | VALID | Retrieval and actions execute through typed registries where bounded; final output is constructed and validated as `FinalAgentOutput`; model-selected ReAct/planner tools now causally determine the observations and executed tools. |
| Timeout retries could overlap unfinished work | VALID | Retry attempts are sequential and write tools never retry. Component-specific timeout fault injection verifies detection without concurrent duplicate side effects. |
| Data contracts and synthetic/agent Gold facts were disconnected | VALID | Source-class contracts now run during staging with quarantine outputs and metrics. Spark rebuild loads the development benchmark and populates case, prediction, agent, exploded tool-call, and review facts. |
| Phase/download state mutations could lose concurrent updates | VALID | Read-modify-write transactions now execute under file locks. |
| Artifact records became stale after trace rewrites | VALID | Each producer re-registers final outputs; a repository-wide manifest verifier and tamper test were added. |
| Strict whole-tree typing did not match CI claims | VALID | Full strict mypy now checks all 69 source files in CI and locally. |

## Methodology review

| Finding | Classification | Resolution |
|---|---|---|
| AG3 was not ReAct and AG4/AG5 plans did not drive execution | VALID | ReAct and planner baselines now first choose typed tool requests without evidence, receive only selected tool observations, and make a second final decision. Each architecture has an isolated action ledger. |
| P26 lacked a less-constrained LLM comparator | VALID | The frozen final protocol evaluates AG0, unrestricted AG3, and AG6 on identical cases; primary paired inference compares AG6 with AG3. |
| Benchmark templates, OOD, and duplicate controls were weak | VALID | Benchmark v4 uses eight disjoint surface families, OOD-only entities/features/control-policy combinations, and normalized cross-split duplicate checks. Deterministic labels remain synthetic and are explicitly limited to construct-validity claims. |
| Point-in-time removal used the target label | VALID | AB10 now substitutes a declared future-only failure feature, while AG6 verifies feature event and system-known timestamps. P07 remains the direct bitemporal as-of/leakage experiment. |
| Agent/HITL sample and uncertainty were inadequate | VALID | Validation uses 80 cases balanced over eight scenario types. HITL uses 1,000 paired case/reviewer resamples and reports residual-risk intervals. Smoke-scale external-validity limitations remain explicit. |
| Interaction rows were not factorial contrasts | VALID | P23 now computes difference-in-differences contrasts from full, each single removal, and the double removal, with paired bootstrap intervals. |
| Retrieval was a 12-query smoke test without real CFR | VALID | P14/P15 now use 100 independent NIST assessment-objective queries plus 20 official CFR queries, official CFR temporal versions, NIST controls, and SEC distractors. |
| Validation and OOD anomaly results were pooled; semi-supervised variants overlapped | VALID | U0-U3 now emit separate validation and OOD metrics. One-pass pseudo-labeling and confidence-filtered iterative self-training are distinct and reuse identical labeled subsets. |

## Security review

| Finding | Classification | Resolution |
|---|---|---|
| P19 did not attack the integrated workflow | VALID | Every S01-S15 row now executes the governed workflow and its relevant real component boundary, with a separate benign control. |
| Correctly hashed poisoned retrieval could bypass narrative-only scanning | VALID | Injection detection scans retrieved content as well as the case; the suite includes a correctly hashed poisoned evidence item. |
| Human approvals were caller-asserted and not durably bound | VALID | Human issuance requires a provisioned reviewer entitlement. A typed decision is persisted against the exact action hash and must match the signed reviewer identity, role, scope, policy, evidence, and decision before execution. |
| Identity purpose/region/session were not policy inputs | VALID | The policy rejects missing or unentitled purpose, non-US region, empty session, business-unit mismatch, clearance failure, and recursive identity-argument tampering. |
| SQL denied-function checks were ineffective | VALID | Function names are normalized from the AST; reflective and file-input functions have parameterized rejection tests. |
| Audit chain was unkeyed and rollback absent | VALID | Action events are hash chained, externally HMAC-anchored, truncation tested, and support recorded simulated rollback. |
| Release gates ignored P19 and missing metrics could promote | VALID | P26 incorporates security-suite S07/S08/S09/S12/S14/S15 outcomes, and any missing/non-finite gate value produces `NO_PROMOTE`. |
| Post-search authorization API exposed restricted content to scorers | VALID | The post-filter API was removed; governed retrieval constructs the authorized point-in-time partition before building the index. |

The mandatory P24 gate remains open. A new clean commit must be reviewed independently by correctness, methodology,
and security reviewers after all validation artifacts finish.
