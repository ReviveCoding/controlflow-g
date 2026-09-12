# P24 Repeat-17 Review Dispositions

Review snapshot: `aad10224f0c76212a5496c783edd4554967155b3`.

The correctness and security reviewers independently returned `PASS`. The
methodology reviewer returned `FAIL` with one HIGH finding. No reviewer opened
the locked holdout or invoked P25/P26. The main agent verified the versioned
ledger and journal evidence and classified the finding `VALID`.

| ID | Reviewer | Severity | Disposition | Repair and validation |
|---|---|---:|---|---|
| R17-M1 | Methodology | HIGH | VALID | AG2-AG5 traces had been preserved from an older runtime while AG6 was regenerated after the cleanup-before-publication repair. All AG0-AG6 architectures were rerun together under current source using fresh `protocol17` ledgers and `graph_state_v11`. The resulting 560 traces contain exactly 80 common cases per architecture with no duplicates. Fresh audit counts are AG2=160, AG3=43, AG4=80, AG5=80, and AG6=720, with no worker-capacity or admission-quarantine errors. |

P18, P21 business metrics, and P23 validation statistics were regenerated from
the full current-source run. The trace SHA-256 is
`81e0a1d86e143eed753e7d3ccde4da1a6a29c05b64b80fe543fa7d982bec1580`.
The negative comparison was retained: AG5 STC is `0.0375` and AG6 STC is
`0.0`.

All work remained pre-freeze. `sealed_test_consumed` was false, P25/P26 were
not invoked, and no locked final IDs, rows, statistics, or outcomes were
opened.
