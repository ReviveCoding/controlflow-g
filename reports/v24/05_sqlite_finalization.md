# SQLite finalization

The V2.4 executor/evaluator SQLite connections use explicit ownership and deterministic close. After the one-shot run, the dedicated finalizer recorded `wal_checkpoint(TRUNCATE) = (0,0,0)`, SQLite integrity `ok`, 600 valid ledger events, and exit code 0 (`results/v24/qualification/sqlite_finalization.json`). The separate hash-only process found identical size and SHA-256 after the frozen two-second interval (`results/v24/qualification/sqlite_stability.json`). These successes did not make qualification admissible: the later post-close verifier failed.
