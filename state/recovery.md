# Recovery Runbook

1. Read `state/execution_state.json`, then validate the last artifact named there against `state/artifact_manifest.json`.
2. Read the journal tail and resume the current phase from its idempotent phase marker; never delete valid prior outputs merely because a phase is rerun.
3. If `state/gpu.lock` exists, verify its recorded PID is alive before treating the GPU as occupied. Remove only a demonstrably stale lock and journal that recovery.
4. Keep `sealed_test_consumed=false` and final labels inaccessible until P25 has produced a deterministic freeze hash.
5. On partial writes, retain the completed target and discard only the phase-owned `.tmp` file after checking its path is inside this repository.
6. After a material repair, rerun affected validation experiments and reviews. After P26, never reuse the same holdout following an invalidating repair.
7. Record every resume, failure, recovery, and state transition in `state/execution_journal.jsonl`.
8. The selected local trust root is `state/audit_trust`, separate from mutable ledger artifacts. Before final seal consumption, run the built-in atomic create/fsync/replace/readback probe. A recovery verifier is pinned when the ledger is bootstrapped; callers may supply only an action-bound approval token, actor, and reason.
9. Local same-host protection is a production-like simulation boundary, not KMS assurance. A real deployment must replace it with a separately administered service identity plus ACL/KMS or an append-only external audit service.
10. P26 uses `state/final_run.json` to bind the one consumed holdout to the exact freeze hash and deterministic run ID. Resume only that run. Before seal consumption and before each work item, write the protected append-only `final-progress-*` journal transition. Each completion stores its full authenticated record in that protected journal before creating the workspace mirror. Reconstruct a deleted mirror from the protected record; never recompute protected completed work. An interrupted active work ID is the only work eligible to resume.
11. Audit recovery approvals use the distinct `state/audit_trust/recovery-approval.key`. The runtime pins its verifier at ledger construction. The root signing key and recovery key are provisioned under trust-root-wide locks with exclusive creation and length validation.
12. Before final synthesis, verify the exact checkpoint work set, progress-chain head, and protected checkpoint attestation. P27–P30 must read final result/trace files once into immutable bytes, verify those exact bytes against the protected final-result attestation, and parse only the verified buffers.
## 2026-09-12 repeat-10 pre-final repair

- Final holdout remains sealed (`sealed_test_consumed=false`); P25/P26 were not invoked.
- Repeat-10 correctness, methodology, and security findings were all classified `VALID`; see `reports/reviews/p24_repeat10_dispositions.md`.
- Protected progress now recovers a fully verified contiguous entry/head crash window and rejects gaps or divergent successors.
- Tool deadlines now cancel a cooperative lease, do not overlap retries, and are checked at the action-ledger commit boundary. A delayed-write regression test proves no late commit.
- Invocation contexts are bound to the exact canonical ordered tool plan and arguments.
- AG6 now consumes governed risk, anomaly, structured, and retrieval context; generated analysis is bounded into structured fields and hashed per trace.
- System tool-argument accuracy and LLM-plan argument accuracy are separate, with no-tool runs reported as N/A.
- P16/P17 were regenerated (560 unique traces), P19 was regenerated (15/15 attacks blocked), P20 was regenerated (12/12 recoveries), P22 was regenerated (1,200 unique traces), and P23/business/HITL artifacts were regenerated.
- Validation AG6: STC `0.10`, nominal task success `0.1375`, 80/80 structured analyses valid after bounded normalization, and system tool-argument accuracy `1.0`.
- Full checks: Ruff pass, strict mypy pass (74 source files), pytest 66 passed, sdist/wheel build pass, artifact manifest verification returned no mismatches.

## 2026-09-12 repeat-11 pre-final repair

- Final holdout remains sealed (`sealed_test_consumed=false`); P25/P26 were not invoked.
- Repeat-11 correctness, methodology, and security BLOCKER/HIGH findings were all classified `VALID`; see `reports/reviews/p24_repeat11_dispositions.md`.
- Governed LLM analysis now requires an exact typed schema and citations restricted to authorized retrieved evidence; every parse/schema/support failure is fail-closed as `INSUFFICIENT_EVIDENCE`.
- Validation AG6 now records the adverse result honestly: STC `0.0`, nominal task success `0.1375`, and 0/80 strict analyses valid/supported.
- P22 cache identity covers the complete visible governed context, and tool-argument accuracy includes rejected/failed authoritative attempts.
- P26 has a run-wide interprocess owner lock; invocation contexts are single-use and unknown identifiers fail closed.
- Detached timeout workers were removed. Tool calls are synchronous under cooperative deadlines, and action/rollback commits use the deadline commit permit.
- Tests use an injected temporary corpus root and do not bootstrap canonical repository artifacts.
- P19 regenerated with 15/15 programmed attacks blocked, P20 with 12/12 injected failures recovered and no duplicate execution, P22 with 15 configurations and 1,200 unique traces, followed by P21/P23 regeneration.
- Artifact registration and exact clean-export validation are required before repeat-12 review.
- The first exact export exposed a scheduler-sensitive commit-race test setup; its pre-commit window was made deterministic and passed five consecutive runs without changing runtime behavior.
