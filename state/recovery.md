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
