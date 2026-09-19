# Fault injection

The V2.4 tests cover open connections, reader/writer checkpoint blocking, missing database, tampered ledger, injected finalizer crashes, WAL frames, byte mutation between stability samples, denominator null behavior, GPU-owner mismatch, decision-artifact mutation, and terminal-manifest mutation (`tests/v24/`). The final admitted static/test gate recorded 195 passing tests in `state/v24_static_quality.json` and `results/v24/v24_tests.xml`. The frozen post-close checkpoint-field bug was not covered and is a test-coverage limitation.
