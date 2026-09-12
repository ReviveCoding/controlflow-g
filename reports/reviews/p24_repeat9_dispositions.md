# P24 Independent Review — Repeat 9 Dispositions

Snapshot reviewed independently by correctness, methodology, and security reviewers: `93cb0093b168f3a424c4f524810d9ccb1fceea12`. All reviewers confirmed `locked-test-accessed=false`; P25/P26 remained blocked.

| Finding | Class | Main-agent classification and repair |
|---|---|---|
| Completed final checkpoints could be deleted before the final checkpoint-set head | BLOCKER | `VALID`: a protected append-only progress chain now records initialization before seal consumption, active work before evaluation, and the full authenticated record plus cumulative record map at every completion. Resume verifies the complete chain, rejects head rollback/replay/deletion, reconstructs missing workspace mirrors from protected records, and permits only the protected active work to resume. |
| P27–P30 verified mutable files and then reopened them | HIGH | `VALID`: final files are now read once into bytes, those exact bytes are hash-verified, and every downstream consumer parses only the immutable verified buffers. |
| Mutable final-run runtime/retry fields could become attested evidence | MEDIUM | `VALID`: attempt timestamp, accumulated runtime, and retry count now advance inside the protected progress journal; final scoring no longer trusts the mutable descriptor for these fields. |
| AG1/AG2 included hidden non-causal tool latency/cost | HIGH | `VALID`: AG1 receives no tool context; AG2 invokes and reuses only the two retrieval outputs visible to its model. P16/P17 and dependent P22/P23 artifacts must be regenerated. |
| Planner order/duplicates were collapsed before observations | MEDIUM/HIGH | `VALID`: ordered `(tool, arguments)` calls are preserved, observations carry step IDs, and duplicate/malformed plans are rejected before any implementation runs and again at execution. |
| Risk/anomaly selection executed an unselected sibling tool | HIGH | `VALID`: risk and anomaly request paths are independent and reuse only their exact selected output. |
| Final checkpoint trace schema omitted downstream-consumed fields | MEDIUM | `VALID`: the strict, extra-forbidden schema now types every field used by P26 gates and P28 statistics. |
| Declared tool timeout blocked indefinitely in executor shutdown | HIGH | `VALID`: the caller now uses a daemon-isolated local worker and bounded queue wait, returns at the declared deadline, never retries writes, and relies on the idempotent action ledger for delayed-write reconciliation. A never-returning regression asserts wall-clock enforcement. This is a local simulation boundary, not process-level tenant isolation. |
| Same-case/session concurrent contexts could overwrite each other | MEDIUM | `VALID`: every context now has a cryptographically random invocation ID and separate cache key. Explicit IDs bind consumption; ambiguous legacy consumption fails closed. |

Nonsealed repair verification initially passed Ruff, strict mypy, 63 tests, and focused attestation/tool-boundary regressions. A new clean-snapshot three-way review is required after affected validation artifact regeneration.
