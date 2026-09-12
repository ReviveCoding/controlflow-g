# P24 Repeat-10 Review Dispositions

Review snapshot: `fef6f19cb077c1022407bd630cff267767b8a1a4`.

The correctness, methodology, and security reviewers independently returned
`FAIL` without opening the locked final holdout. The main agent reproduced
and classified every material finding as `VALID`. No finding was dismissed as
invalid or not applicable.

| ID | Reviewer | Severity | Disposition | Repair and validation |
|---|---|---:|---|---|
| R10-C1 | Correctness | HIGH | VALID | CI no longer references an absent `scripts` directory. A deterministic test-session bootstrap creates only the minimal staging corpus and hash manifest when a clean checkout lacks runtime data. Exact-checkout validation is repeated before the next review. |
| R10-C2 / R10-S2 | Correctness, Security | HIGH | VALID | Tool calls now receive a cooperative deadline lease. Timeout cancels the lease, records `timeout_indeterminate`, and never overlaps a retry. The action ledger rechecks the lease immediately before every write commit. A regression test proves a delayed write cannot commit after timeout. |
| R10-C3 | Correctness | HIGH | VALID | Each invocation context stores its canonical ordered tool plan and exact arguments. Execute-time submissions must match before cached observations can be consumed. An altered-arguments/same-context regression test fails closed. |
| R10-M1 | Methodology | HIGH | VALID | AG6 exposes calibrated risk, anomaly, structured evidence, and retrieval output to a governed LLM-analysis turn. The generated analysis is normalized into bounded structured fields, hashed per trace, and contributes to structured-output validity. A tiny-model JSON-key failure observed during validation is retained in the work log; the adapter now bounds nonempty generated text instead of fabricating a semantic field. P16/P17 and dependent experiments are regenerated. |
| R10-M2 | Methodology | HIGH | VALID | After irreversible P26 seal consumption but before evaluation work, raw split identifiers and holdout case identifiers must each be nonblank and unique, have identical counts, and form an exact set bijection. Silent set collapse/intersection was removed. |
| R10-M3 | Methodology | HIGH | VALID | Traces now persist ordered successful tool steps and exact arguments. `tool_argument_accuracy` scores system execution, `llm_plan_argument_accuracy` separately scores agent plans, and no-tool cases are N/A rather than false. Dependent results are regenerated. |
| R10-S1 | Security | HIGH | VALID | The protected progress reader accepts only a signed, contiguous successor chain extending the authenticated head, then advances the head after full verification. Gaps, deletions, and divergent signed entries remain fail-closed; both recovery and rejection paths have regression tests. |

All changes remain pre-freeze. `sealed_test_consumed` is false, P25/P26 have
not been invoked, and no locked final IDs, rows, statistics, or outcomes were
opened during remediation.
