# P24 repeat-2 dispositions

Review snapshot: `ff6f944648ad2267fec4197a339ee1e87d90c06b`.
The snapshot failed and was not frozen. The locked test was not accessed.

## Correctness review

- COR-R2-01 — **VALID / repaired**. AG6 now executes the authorized point-in-time
  corpus through sparse+dense hybrid retrieval and the pinned cross-encoder.
  Risk, anomaly, control/regulation retrieval, historical-case search,
  point-in-time policy lookup, structured case/transaction queries, evidence
  bundling, and the simulated write are registered typed tools.
- COR-R2-02 — **VALID / repaired**. A durable workflow wrapper now checkpoints
  before execution, persists completed traces, resumes in a new process, and
  relies on the action ledger to make a crash after execution safe to replay.
  P20 was rerun against distinct model, retriever, and SQL adapter boundaries.
- COR-R2-03 — **VALID / repaired**. The GPU semaphore now uses an OS-backed
  interprocess file lock and owner token; stale owner metadata is never removed
  before acquiring the actual lock.

## Methodology review

- METH-R2-01 — **VALID / repaired**. Training/validation use disjoint narrative
  families, the temporal split is a strict chronological tail, and the entity
  split has zero training-entity overlap. P09 and all affected experiments were
  regenerated.
- METH-R2-02 — **VALID / repaired**. The unconsumed holdout was replaced by a
  separately generated 2025 artifact with a cryptographically random,
  undisclosed seed commitment, holdout-only templates, IDs, and entities. It is
  no longer reconstructible from the public development seed or split code.
- METH-R2-03 — **VALID / repaired**. P22 now executes every ablation end to end
  and regenerates an LLM output whenever the byte-identical context cache misses.
- METH-R2-04 — **VALID / repaired**. Risk-coverage records Wilson 95% bounds and
  threshold eligibility uses the lower critical-capture and upper residual-risk
  bounds. Insufficient evidence fails closed to a threshold of 1.01.
- METH-R2-05 — **VALID / repaired**. P09 now produces feature-source IDs and
  matched event/system timestamps through `build_case_features`; governed risk
  uses the resulting `pit_historical_failures` value.
- METH-R2-06 — **VALID / repaired**. ReAct/planner arguments now pass through
  each typed input schema and authorization boundary, affect execution, and are
  scored separately.
- METH-R2-07 — **VALID / repaired**. M0/AG0 now uses an independently authored
  threshold heuristic rather than the generator coefficients.
- METH-R2-08 — **VALID / repaired**. Validation and final confidence intervals
  use entity-cluster resampling; all primary and interaction p-values receive a
  Holm adjustment.
- METH-R2-09 — **VALID / limitation retained**. Retrieval includes independently
  authored NIST queries but assessment-objective and CFR queries remain a
  controlled source-grounded benchmark. Reports must not claim broad retrieval
  generalization.

## Security review

- SEC2-01 — **VALID / repaired**. P19 records an attack-specific causal
  invariant and matched benign boundary for S01–S15. Blocking is no longer
  inferred from a generic lack of action. S10, S13, and S14 feed zero-tolerance
  final security aggregation.
- SEC2-02 — **VALID / repaired**. Verifier support requires trusted ingestion
  plus an explicit claim relation for every required item. Correctly hashed
  untrusted `[SUPPORTS]` text is rejected and regression-tested.
- SEC2-03 — **VALID / repaired**. Writes require exact `ALLOW`, use the actual
  authorization outcome in the signed system token, carry protected data
  classification, and reauthorize immediately before execution. External IdP
  integration remains outside this local simulation and is documented.
- SEC2-04 — **VALID / repaired**. Rollback requires an entitled reviewer token
  bound to the exact target action, reason, policy version, and evidence hash.
- SEC2-05 — **VALID / repaired with deployment limitation**. Every audit event
  binds the complete materialized action row and verification cross-checks the
  latest event against current state. The local key/head is not an external WORM
  anchor; production integration remains a documented requirement.
- SEC2-06 — **VALID / repaired**. Durable state filenames are SHA-256 case keys,
  eliminating case-ID path traversal.

All quantitative artifacts affected by these repairs must be regenerated before
the next review snapshot. This document records main-agent dispositions, not a
review pass.
