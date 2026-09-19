# Schema mutation tests

`tests/v25/test_checkpoint_contract.py` rejects missing or wrongly nested fields, wrong types, absent hashes, truncated JSON, unknown schema versions, and incompatible nested fields. It also tests frozen threshold, bundle, gate, and freeze-hash drift. `tests/v25/test_rehearsal_evidence.py` loads the real producer checkpoint and checks the terminal manifest and unbound-state mutation controls.

`state/v25_static_quality.json` records passing `ruff format --check`, `ruff check`, strict mypy on the active source/scripts, and 220 passing full-suite tests. The JUnit file is `results/v25/v25_tests.xml`.
