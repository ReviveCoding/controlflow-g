# SQLite regression

The V2.4 explicit owned-connection lifecycle, dedicated finalizer subprocess, `wal_checkpoint(TRUNCATE)`, and hash-only stability verifier were retained. In the 25013 rehearsal, `results/v25/development/rehearsal_25013/sqlite_finalization.json` records successful finalization and `results/v25/development/rehearsal_25013/sqlite_stability.json` records stable post-close hashes. The finalizer recorded no busy checkpoint and no remaining WAL/SHM sidecars.
