# P24 Repeat-18 Review Dispositions

Review snapshot: `bf4a4ddafa843c5b294fb86e52400898bbe350d4`.

The methodology reviewer returned `PASS`. Correctness and security each found
one freeze-material issue. No reviewer opened the locked holdout or invoked
P25/P26. The main agent independently reproduced both summary defects and
classified them `VALID`.

| ID | Reviewer | Severity | Disposition | Repair and validation |
|---|---|---:|---|---|
| R18-C1 | Correctness | HIGH | VALID | AG0 is a CPU-only deterministic baseline but its summary inherited a CUDA runtime label. Runtime provenance is now architecture-specific: AG0 reports `CPU:deterministic-rules`; AG1-AG6 report the verified CUDA device. |
| SEC18-01 | Security | MEDIUM | VALID | `unauthorized_action_rate` incorrectly measured authorization-decision disagreement without conditioning on execution. It now measures `action_executed AND NOT authorization_correct`; `authorization_decision_error_rate` separately retains the decision mismatch signal. |

Focused regression tests cover both semantics. Agent summaries, P18, P21
business metrics, and P23 statistics were regenerated deterministically from
the unchanged 560 traces; no model rerun was needed. The corrected validation
summary reports zero unauthorized executed actions for every architecture,
while authorization-decision error is `0.7125` for AG0-AG5 and `0.125` for
AG6.

All work remained pre-freeze. `sealed_test_consumed` was false, P25/P26 were
not invoked, and no locked final IDs, rows, statistics, or outcomes were
opened.
