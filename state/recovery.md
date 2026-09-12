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
