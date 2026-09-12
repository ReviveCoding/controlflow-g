# P24 Repeat-11 Review Dispositions

Review snapshot: `1f43a215db622cb3a3536e498fd303426a23ba2e`.

The correctness, methodology, and security reviewers independently returned
`FAIL` without opening the locked final holdout. The main agent reproduced or
code-traced every BLOCKER/HIGH finding and classified it `VALID`. No finding was
dismissed as invalid or not applicable.

| ID | Reviewer | Severity | Disposition | Repair and validation |
|---|---|---:|---|---|
| R11-M1 | Methodology | HIGH | VALID | The governed LLM boundary now requires an exact Pydantic schema, enumerated severity/disposition/action values, and nonempty citations drawn only from authorized tool evidence. Parse, schema, or evidence-support failure is fail-closed as `INSUFFICIENT_EVIDENCE`; raw text is never promoted into a semantic field. The regenerated AG6 validation result is intentionally adverse: 0/80 strict analyses valid and STC `0.0`. |
| R11-M2 | Methodology | HIGH | VALID | P22 context caching now binds the complete model-visible governed context (except the one-time context identifier), including risk, anomaly, structured evidence, retrieval output, and tool observations. Controls/regulations alone no longer determine cache reuse. All 15 ablations and 1,200 traces were regenerated. |
| R11-M3 | Methodology | HIGH | VALID | Traces persist every authoritative attempted tool step, including rejected or unexecuted attempts. Generic tool-argument accuracy scores those attempts; invalid attempts are zero and only genuine no-attempt cases are N/A. AG3 now has 43 attempted cases scored zero and 37 genuine no-attempt cases. |
| R11-C1 | Correctness | BLOCKER | VALID | P26 now holds a process-wide interprocess owner lock across the entire one-shot final run, with a freeze/run-bound owner record and takeover history. A concurrent owner fails before seal or evaluation work. |
| R11-C2 / R11-S1 | Correctness, Security | HIGH | VALID | Detached timeout workers were removed. Tool implementations execute synchronously under a cooperative deadline, and all irreversible ledger commits acquire the deadline commit permit. A call that finishes after its deadline is rejected without any later worker; a commit already begun before expiry completes synchronously and is audited. Reliability and race regressions were rerun. |
| R11-C3 | Correctness | HIGH | VALID | Unknown or already-consumed invocation context identifiers now fail closed. A context can be used only once and only with its exact canonical ordered plan and arguments. |
| R11-C4 | Correctness | HIGH | VALID | Test-session bootstrap no longer writes canonical repository data or artifact-manifest paths. Tests receive an explicitly injected temporary corpus root; the next clean-export verification checks that canonical paths remain absent before and after the suite. |
| R11-C5 | Correctness | HIGH | VALID | Recovery documentation and all changed result/report artifacts are re-registered after the repeat-11 record is written, so manifest verification is performed against the final bytes rather than stale hashes. |

P19 regenerated with all 15 programmed attacks blocked and their causal
invariants satisfied. P20 regenerated with all 12 injected failures recovered
and no duplicate execution. P21 and P23 were regenerated after P22. These are
validation results, not final-test outcomes.

The first exact-export test exposed a scheduler-sensitive 10 ms setup window in
the commit-race regression. The runtime was unchanged: the test now allows a
200 ms pre-commit window and deliberately holds the commit beyond it. Five
independent repetitions pass; the distinct delayed-write rejection test keeps
its 10 ms deadline.

All changes remain pre-freeze. `sealed_test_consumed` is false, P25/P26 have
not been invoked, and no locked final IDs, rows, statistics, or outcomes were
opened during remediation.
