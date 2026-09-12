# P24 Repeat-13 Review Dispositions

Review snapshot: `86b04a654c6af18457bb35edc3733ec5b5ae0170`.

The correctness, methodology, and security reviewers independently returned
`FAIL` without opening the locked holdout. The main agent reproduced or
code-traced every material finding and classified it `VALID`.

| ID | Reviewer | Severity | Disposition | Repair and validation |
|---|---|---:|---|---|
| R13-C1 | Correctness | BLOCKER | VALID | `FinalCheckpointTrace` now includes both `recommended_action_correct` and `root_cause_correct` plus the bounded root-cause text. A regression constructs an actual `evaluate_trace()` record and validates it through the strict, extra-forbid P26 checkpoint schema before any seal access. |
| R13-C2 / R13-S2 | Correctness, Security | HIGH | VALID | Tool workers use a process-wide one-slot admission circuit. A timed-out worker retains the sole slot until it exits, so no later tool worker can overlap it; admission fails closed as `worker_capacity_exhausted`. A regression holds the worker, proves the next invocation is rejected without spawning another worker, releases it, and confirms cleanup. This bounded local thread boundary is still not process/service isolation. |
| R13-S1 | Security | HIGH | VALID | The deadline commit permit now covers both the SQLite COMMIT and signed external-anchor synchronization. Commit-zone entry is recorded and completion is set in `finally`, so anchor failure or timeout after DB commit is always classified for reconciliation, never as pre-commit cancellation. |
| R13-M1 | Methodology | HIGH | VALID | Provenance/time/authorization/hash checks are now named `llm_analysis_evidence_valid`, not semantic support. A separate deterministic benchmark predicate scores root-cause correctness using the expected control plus case-type causal concepts and rejects explicit contradictions. Root-cause correctness and expected-action correctness both gate governed STC. Nonsense, contradiction, valid normal, and valid missing-evidence fixtures are tested. |

The recorded governed validation outputs were deterministically rescored rather
than regenerated because every governed analysis had already failed schema
validation. The rescorer asserts this prerequisite and fails closed if any
valid governed claim would require unavailable historical text. It renames the
evidence-validity fields, assigns governed root-cause correctness false,
preserves non-governed outcomes, and re-registers P17/P22 artifacts. Future
runs persist the bounded root-cause text and use fresh ledger/checkpoint
versions.

Validation remains adverse: AG6 root-cause correctness is 0/80, nominal task
success is `0.1375`, and STC is `0.0`; all 15 ablation STC estimates remain
`0.0`. P20/P21/P23 were regenerated after rescoring. These results remain
validation-only.

All changes remain pre-freeze. `sealed_test_consumed` is false, P25/P26 have
not been invoked, and no locked final IDs, rows, statistics, or outcomes were
opened during remediation.
