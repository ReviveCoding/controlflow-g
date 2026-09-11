# P24 Independent Review — Repeat 4 Dispositions

Snapshot reviewed: `48fde12b3350489735263b7fa513e2bb2762eda0`.

The methodology reviewer passed the snapshot. Correctness and security findings below were independently validated by the main agent and classified `VALID`. P25 remained blocked; the locked test was not accessed.

| Finding | Class | Disposition and repair |
|---|---|---|
| Historical-case, policy-at-time, structured-case, transaction, and evidence-bundle tools were non-causal | BLOCKER | `VALID`: all five now execute governed queries and their typed results enter the LLM-visible context and durable downstream state. Historical cases use lexical similarity, transactions use entity/time-filtered staging records, policy results are time-valid retrieved records, and bundles hash retrieved IDs. |
| Authorization identity was synthesized from mutable case fields | HIGH | `VALID`: a trusted session identity provider now resolves immutable identity from an opaque session token; workflow entry points require the token. A regression proves changing case business unit does not change authenticated identity. |
| External anchors were not concurrency/crash consistent | HIGH | `VALID`: ledger-wide interprocess serialization covers database-plus-anchor operations; commits precede signed checkpoint export; action/tool writes fail closed on chain failure; durable resume explicitly reconciles the diagnosed commit/export crash window. Concurrent, tamper, and recovery regressions were added. |
| Security interventions were not attack-specific end to end | HIGH | `VALID`: S08 uses recursive identity-argument escalation, S09 restricted-scope exfiltration, S12 case-ID argument substitution, S14 actual action-bound token verification, and S15 claimed-role spoofing. Each has an integrated matched benign workflow path; security status requires component and integrated invariants. |
| Tool/SQL audit events omitted reconstructive fields | HIGH | `VALID`: durable events now include canonical argument/query hash, scope, classification, severity, policy and schema versions, authorization reasons, output hash, status, attempt, and duration. SQL validation emits a durable hashed event. |

Post-repair validation evidence:

- Ruff, strict mypy, and 49 tests passed before the final experiment reruns.
- Integrated security: 15/15 attacks blocked, all causal invariants satisfied, zero matched-control false-positive blocks.
- Recovery: 11/11 injected faults detected and recovered; zero duplicate executions.
- Agent validation (`n=80`): AG6 nominal task success 0.1375, STC 0.1000, evidence coverage 0.1250, P95 latency 3.992108 seconds.
- End-to-end ablations (`n=80` each): full STC 0.1000; no verifier 0; no point-in-time features 0; no ML risk 0.0375; no authorization and unrestricted tools 0.0625; other tested removals included null effects.

A repeat-5 independent review is required on a new clean snapshot before P24 completion.
