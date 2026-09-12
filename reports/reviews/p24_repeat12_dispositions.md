# P24 Repeat-12 Review Dispositions

Review snapshot: `170ab59326740f220cda1ae8c2531fc98f5827c4`.

The correctness, methodology, and security reviewers independently returned
`FAIL` without opening the locked holdout. The main agent reproduced or
code-traced every material finding and classified it `VALID`. No finding was
dismissed as invalid or not applicable.

| ID | Reviewer | Severity | Disposition | Repair and validation |
|---|---|---:|---|---|
| R12-C1 / R12-S1 | Correctness, Security | HIGH | VALID | Tool implementations again execute in a bounded daemon worker. On deadline, cancellation serializes against the commit permit: pre-commit work returns bounded and can never enter a governed commit; an already-entered atomic commit is awaited, marked `timeout_after_commit_reconciled`, and every later commit is cancelled. Timeouts are never retried. A releasable never-returning regression proves bounded return, and P20 was regenerated. The local thread isolation is not represented as production tenant isolation. |
| R12-C2 / R12-M1 | Correctness, Methodology | HIGH | VALID | The prompt now says exactly five keys and matches the strict model. A cross-field validator allows empty citations only for `INSUFFICIENT_EVIDENCE`, requires an uncited outcome to request evidence, and rejects duplicate citations. P16/P17 were regenerated. The tiny pinned LLM still produced 0/80 strictly valid AG6 analyses; this adverse result is retained rather than normalized away. |
| R12-M2 | Methodology | HIGH | VALID | Citation membership is no longer called semantic support. The independent verifier checks trusted claim relations, temporal and authorization validity, hashes, explicit control/regulation mention, and action/disposition coherence. Evaluation separately scores the benchmark-derived expected recommendation, and governed STC requires checked support plus recommendation correctness. |
| R12-M3 | Methodology | HIGH | VALID | Governed prediction exposes explicit schema-validation and evidence-validation modes. AB06 and verifier interactions disable support checking; AB09 disables strict schema validation while retaining verification. Both flags are part of prediction-cache identity, so removed components cannot reuse the full system prediction. P22/P23 were regenerated with fresh version-14 ledgers/checkpoints. |

Validation-only outcomes after repair:

- P16/P17: 560 traces, 7 architectures, 80 cases each, zero duplicate experiment/case keys.
- AG6: nominal task success `0.1375`, STC `0.0`, strict analysis validity `0/80`, independent support `10/80`, and expected recommendation correctness `10/80`.
- P19: all 15 programmed attacks blocked with causal invariants satisfied.
- P20: all 12 injected failures recovered with zero duplicate executions; 5 ms timeout injections returned in approximately 16 ms including worker/audit overhead.
- P22: 15 summaries and 1,200 traces with no duplicate keys. All STC estimates are `0.0`; this floor/null ablation result is retained. Support checking is disabled only in the three intended no-verifier configurations.
- Ruff passes, strict mypy passes for 74 source files, and pytest passes 71 tests.

All changes remain pre-freeze. `sealed_test_consumed` is false, P25/P26 have
not been invoked, and no locked final IDs, rows, statistics, or outcomes were
opened during remediation.
