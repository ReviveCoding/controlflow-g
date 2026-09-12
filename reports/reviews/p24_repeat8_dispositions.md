# P24 Independent Review — Repeat 8 Dispositions

Snapshot reviewed: `c15483ef2b0b983909c32ea5f1268b264c121e74`.

All methodology, correctness, and security BLOCKER/HIGH findings were independently reproduced or code-traced by the main agent and classified `VALID`. The security MEDIUM about local symmetric recovery authority was also `VALID` as a disclosed production-like simulation limitation. P25 remained blocked and no locked-test rows were accessed.

| Finding | Class | Disposition and repair |
|---|---|---|
| P26 checkpoint records were forgeable and work membership was weakly validated | BLOCKER | `VALID`: P26 now writes one atomically replaced HMAC-authenticated record per work ID under a run-specific directory. Resume verifies the protected signature, filename/work binding, exact run/freeze, trace schema, case/experiment identity, duplicates, and expected work membership. Final synthesis requires the exact full work set and writes a protected checkpoint attestation containing every record digest. |
| Completed final results lacked an independent protected attestation | HIGH | `VALID`: the protected trust root now stores a signed final-result attestation binding run ID, freeze hash, checkpoint-head digest, final result hash, and final trace hash. P27, P28, P29, and P30 verify this attestation before consuming final artifacts. |
| Crash after action commit but before checkpoint changed recovered traces | BLOCKER | `VALID`: workflow traces distinguish achieved action state from `performed_this_invocation`. An idempotent replay of an already executed action reports the same achieved state while correctly reporting no duplicate side effect. A regression executes the workflow twice and asserts equivalent achieved state with exactly one performed side effect. |
| Completion ordering could leave P26 complete outside the phase state | HIGH | `VALID`: P26 now completes and registers the phase before signing and marking the final run complete. If a crash occurs before final-run completion, the authenticated checkpoints reconstruct synthesis; if the run is complete but phase state is stale, the completed branch verifies attested outputs and reconciles P26 registration. |
| Runtime/retry duration was underreported after resume | MEDIUM | `VALID`: the run descriptor persists attempt start, accumulated wall time, and retry count. Each same-freeze resume accounts for the prior interrupted attempt before starting the next. |
| Selected tools executed once for observation and again for evaluation | MEDIUM | `VALID`: authorized typed outputs and evidence are cached by case/session, used in the LLM observation, then reused by workflow scoring/audit state. Cache entries are removed after execution. |
| Tool arguments did not causally determine resources and argument accuracy was schema-only | HIGH | `VALID`: retrieval queries now determine the searched text; every case-scoped tool rejects a case ID different from the active authorized case; duplicate requested tools are rejected as argument errors; deterministic argument accuracy requires exact expected query/case/action arguments. |
| Local recovery verifier also has symmetric signing authority | MEDIUM | `VALID_LIMITATION`: this remains explicitly labeled a local production-like simulation. A real deployment requires verifier-only public-key validation and separately authenticated KMS/HSM/reviewer signing. |

Post-repair validation evidence:

- Ruff, strict mypy, and 60 nonsealed tests passed before experiment regeneration.
- Agent comparison regenerated on fresh ledgers: seven architectures × 80 cases, 560 unique finite traces.
- Ablation regenerated from fresh repeat-8 checkpoints: 15 configurations/interactions × 80 cases, 1,200 unique finite traces.
- Reliability and security suites were regenerated; validation-only paired statistics were regenerated.
- The final holdout remains sealed.

A repeat-9 three-way independent review is required on a new clean snapshot before P24 completion.
