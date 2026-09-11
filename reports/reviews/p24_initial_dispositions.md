# Initial P24 Review Dispositions

The first three-way review was run while repairs were still being written, so it is not the mandatory stable-snapshot review. Its findings were nevertheless independently validated by the main agent. Final test content remained sealed.

## Correctness

| Finding | Classification | Resolution |
|---|---|---|
| Agent, security, reliability and ablation rows encoded outcomes instead of executing controls | VALID | Replaced with observed workflow, guard, fault, ledger and component-removal execution. |
| CFR section parsing and cross-source temporal identity were incorrect | VALID | Repaired parser and canonical bitemporal version windows; rebuilt staging and lakehouse. |
| State and artifact-manifest writes raced | VALID | Added file locks and atomic canonical writes. |
| Verifier conflict/support semantics were reversed or weak | VALID | Rebuilt claim checks around hashes, support markers, conflicts, temporal and authorization validity. |
| GPU model lifetime escaped the semaphore | VALID | Moved model construction, use, attestation and release inside the single-GPU lease. |
| Tool timeouts and action recovery were not executed | VALID | Added enforced timeout/retry traces and append-only, signed, idempotent action execution. |

## Methodology

| Finding | Classification | Resolution |
|---|---|---|
| Agent comparison and ablations were circular/post-hoc | VALID | Same-case observed AG0-AG6 traces and executed AB01-AB12/interaction traces now back summaries. |
| Retrieval queries and labels were circular | VALID | Replaced with independently authored query fixtures and public-corpus relevance truth. |
| OOD cases did not establish entity/style novelty | VALID | Regenerated deterministic benchmark v3 with disjoint OOD entities and style. |
| Point-in-time joins ignored system-known time | VALID | Added dual valid/system-time as-of constraints and late-known tests. |
| Anomaly models selected contamination using evaluation prevalence | VALID | Fixed training-only contamination, scaling and thresholds. |
| Calibration/threshold selection reused evaluation partitions | VALID | Split validation chronologically into calibration-fit, method-selection and threshold subsets; persisted the selected calibrator. |
| Final-test protection and candidate freeze were missing | VALID | Added clean-tree freeze, artifact hashing, once-only fail-closed seal consumption and scorer/gate freezing. |
| Statistical power, latency and cost claims exceeded evidence | VALID | Reports now label the agent sample as smoke-scale and cost as a compute proxy; null/negative results are retained. |

## Security

| Finding | Classification | Resolution |
|---|---|---|
| Fifteen adversarial outcomes were constants rather than attacks | VALID | Each S01-S15 now invokes a distinct observed guard or boundary. |
| AG6 did not integrate retrieval, verifier, policy, HITL and ledger boundaries | VALID | Implemented one typed governed workflow with those boundaries outside the LLM. |
| Approval was a replayable Boolean | VALID | Replaced with HMAC-signed, expiring, action/policy/evidence-bound tokens. |
| Authorization was applied after retrieval | VALID | Role, classification and business-unit partitioning now occurs before scoring/content access. |
| SQL/table/tool arguments were insufficiently bounded | VALID | Added AST-only SELECT validation, qualified allowlists, limits, denied functions and recursive identity-field tamper checks. |
| Release gates were descriptive only | VALID | Added executable, fail-closed gate evaluation with zero-tolerance outcomes. |

All material repairs above were made before candidate freeze. A new three-way review is required against one stable commit; this document does not satisfy that gate.
