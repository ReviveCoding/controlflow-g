# P24 Repeat-14 Review Dispositions

Review snapshot: `006ecfbca1ce71651ad1354ac55479faba500431`.

The correctness and methodology reviewers independently returned `FAIL`; the
security reviewer returned `PASS_WITH_FINDINGS`. None opened the locked
holdout or invoked P25/P26. The main agent reproduced or code-traced every
finding below and classified each `VALID`.

| ID | Reviewer | Severity | Disposition | Repair and validation |
|---|---|---:|---|---|
| R14-M1 | Methodology | HIGH | VALID | Root-cause and recommendation fields are governed-only secondary diagnostics and no longer enter the cross-architecture STC composite. Non-governed architectures receive no synthetic credit for outputs they did not produce. The common STC contract is now identical for every architecture: nominal correctness, evidence, temporal correctness, authorization, structured-output validity, and applicable evidence-validity checks. Existing baseline traces were deterministically rescored from those recorded fields. |
| R14-M2 | Methodology | HIGH | VALID | The free-text keyword scorer was removed. Governed output now has a strict enumerated `root_cause_code`; correctness is exact equality with deterministic benchmark truth. Free-text hypotheses remain bounded evidence text but are not treated as semantic truth. Unknown codes and invalid analyses fail closed. AG6 and all governed ablations were rerun because the prompt/schema changed. |
| R14-C1 | Correctness | HIGH | VALID | Local tool-worker lifetime is registered before thread start and released only in the worker `finally`. `GpuSemaphore` retains its host-wide file lock while any local tool worker remains alive, including after caller timeout. A regression proves a second GPU lease is rejected while the quarantined worker remains, then succeeds after that worker exits. A never-terminating local thread therefore requires process restart but cannot permit advertised GPU overlap. |
| R14-S1 | Security | LOW | VALID | Anchor-specific regressions now block signed-anchor synchronization beyond the caller deadline, prove the caller cannot report reconciliation early, verify the resulting chain, inject anchor failure, and verify the committed-but-unanchored ledger fails closed. |

During affected P20 regeneration, the new circuit exposed an idempotency flaw
in the fault harness: immediate scenarios could run before the timed-out worker
drained, and existing crash checkpoints were misclassified on rerun. Recovery
now explicitly includes quarantine drain time, while crash checkpoints are
durable evidence of prior injection and resume safely. Fresh versioned
artifacts demonstrated 12/12 detected recoveries and zero duplicate execution.

Validation remains adverse. AG6 produced 0/80 strict six-field analyses,
typed root-cause correctness 0/80, nominal task success `0.1375`, and STC
`0.0`. The 15 ablation configurations contain 1,200 paired traces and all STC
estimates remain `0.0`. These are validation-only results.

All repairs remain pre-freeze. `sealed_test_consumed` is false, P25/P26 have
not been invoked, and no locked final IDs, rows, statistics, or outcomes were
opened during review or remediation.
