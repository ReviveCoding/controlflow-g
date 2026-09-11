# P24 Independent Review — Repeat 5 Dispositions

Snapshot reviewed: `6bff861635383ccb9264f6a1ff197d7da8e6ef0a`.

All repeat-5 BLOCKER/HIGH findings were independently reproduced or code-traced by the main agent and classified `VALID`. P25 remained blocked and the locked test was not accessed.

| Finding | Class | Disposition and repair |
|---|---|---|
| Final trust decisions depended on the mutable artifact manifest | BLOCKER | `VALID`: `GovernedWorkflow` accepts verified corpus hashes, and P26 supplies the immutable hash map embedded in the already verified freeze manifest. The live artifact manifest is not a transitive final-evaluation input. |
| Agent latency and GPU proxy omitted context construction | HIGH | `VALID`: context construction is timed from credential resolution through all typed retrieval/model/tool outputs; its CUDA retrieval time is recorded separately and added to LLM inference time. Agent and ablation validation artifacts and paired statistics were rerun. |
| Cross-business-unit reads were possible because resource scope was not bound to identity | BLOCKER | `VALID`: both workflow entry points bind the authoritative case business unit to the authenticated identity before retrieval, feature/model computation, transaction access, or tool registration. Tool policy receives the resource scope. Cross-unit context, execution, and forged-credential tests fail closed. |
| Missing action anchors were silently re-signed on ledger reopen | HIGH | `VALID`: constructor repair was removed. Missing/deleted anchors fail verification and ordinary execution. Recovery is an explicit operation bound to observed database heads and an entitled reviewer’s signed approval, then recorded in the independent system event chain. |
| Session credentials were predictable and callers could request a token by scope | BLOCKER | `VALID`: credentials are high-entropy opaque random values returned once to the bootstrap caller; the provider retains only a private credential-to-identity map and exposes resolution only. Session identifiers are separate non-credential values. S15 now submits an attacker-chosen forged credential through the integrated workflow. |
| Empty-chain truncation and unauthenticated reconciliation could bypass audit integrity | BLOCKER | `VALID`: integrity checks always verify both chains, including an empty database against an existing signed head. Constructor auto-repair is gone. Truncate-then-execute and delete-anchor/reopen regressions fail closed. Reconciliation requires a signed, action/evidence-bound recovery approval and emits a recovery audit event. |
| Audit signing material shared the mutable experiment directory | BLOCKER | `VALID` for the production-like local profile: signing key and signed heads default to user-local AppData outside the repository and ledger directories; the key is created with restrictive mode. The first E:-volume implementation failed closed during validation, leading to the stable AppData backend and configuration-level ablation checkpoints. A production deployment still requires a service-identity ACL/KMS or append-only external ledger; this local same-host limitation is explicit and is not represented as production deployment. |

Post-repair validation evidence:

- Ruff, strict mypy, and 50 nonsealed tests pass.
- Integrated security: 15/15 attacks blocked, all causal invariants satisfied, and zero matched-benign false-positive blocks; S15 exercises a forged credential.
- Recovery: 11/11 injected faults detected and recovered with zero duplicate executions.
- Agent validation (`n=80` per architecture): AG6 nominal task success 0.1375, STC 0.1000, evidence coverage 0.1250, and corrected E2E P95 latency 4.068188 seconds.
- End-to-end ablations: 15 configurations/interactions, `n=80` each, 1,200 traces, no duplicate experiment/case pairs, and finite latency/GPU fields. Full STC is 0.1000; no verifier and no point-in-time features are 0; no ML risk is 0.0375; no authorization and unrestricted tools are 0.0625.
- The final holdout remains sealed (`sealed_test_consumed=false`).

A repeat-6 independent review is required on a new clean snapshot before P24 completion.
