# P24 Independent Review — Repeat 7 Dispositions

Snapshot reviewed: `5b4b48d5631f521ca8ddc9dfd0fa3f27bcba4e3e`.

Security passed. The methodology HIGH, correctness BLOCKER, and two correctness HIGH findings were independently code-traced or reproduced by the main agent and classified `VALID`. P25 remained blocked and no locked-test rows were accessed.

| Finding | Class | Disposition and repair |
|---|---|---|
| Planner/ReAct latency and cost included unselected eager tools | HIGH | `VALID`: planner/ReAct now performs its first model turn against tool schemas only, then invokes only explicitly selected typed tools with the exact model-supplied arguments. Invalid arguments are rejected before implementation and yield bounded error observations. Model-generation timing excludes orchestration/tool wall time; end-to-end latency adds separately measured context/tool time, and retrieval GPU time is charged only when neural retrieval actually executes. |
| P26 seal consumption had no durable resume protocol | BLOCKER | `VALID`: P26 now creates a deterministic freeze-bound run descriptor before consuming the seal and permits only the same run/freeze to resume. Each case/architecture trace is fsynced with a unique work ID before progress continues. Completion requires the full expected checkpoint cardinality, durable Parquet outputs, an output hash, and a completed run descriptor. |
| Runtime ledgers lacked a recovery verifier | HIGH | `VALID`: local runtime, final, security, ablation, reliability, and recovery-worker ledgers now receive a verifier loaded from a distinct persistent recovery-approval trust key. The 12th reliability scenario exercises anchor loss, action-bound approval, reconciliation, and chain verification. |
| Shared audit signing-key bootstrap raced across distinct ledgers | HIGH | `VALID`: root-key provisioning now uses a trust-root-wide interprocess lock, exclusive creation, fsync, and key-length validation. A four-process distinct-ledger clean-start regression confirms all chains verify under the one shared key. |

Post-repair validation evidence:

- Ruff formatting/lint and strict mypy passed; 56 nonsealed tests passed, including freeze-bound final-run resume, exact commit-before-anchor recovery, and multiprocess distinct-ledger bootstrap regressions.
- Agent validation contains 560 unique traces across seven architectures with finite timing/GPU fields. For AG3/AG4/AG5, 43/80/80 traces requested tools, all were rejected for invalid arguments, and zero tool implementations executed.
- Reliability: 12/12 injected faults recovered with zero duplicate executions, including audit-anchor crash reconciliation.
- Security: 15/15 attacks blocked with zero attack successes and zero matched-benign false-positive blocks.
- Validation-only statistics were regenerated. The final holdout remains sealed.

A repeat-8 three-way independent review is required on a new clean snapshot before P24 completion.
