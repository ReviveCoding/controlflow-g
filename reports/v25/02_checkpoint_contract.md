# Checkpoint contract

`src/controlflow/v25/checkpoint_contract.py` owns the physical checkpoint layout and validates schema version 1, top-level completion fields, the nested `fields` object, types, hash formats, and required mappings. Active V2.5 resume and integrity consumers use the typed view. The real 60-case checkpoint is `artifacts/v25/development/rehearsal_25013/rehearsal.checkpoint.json`.

The committed freeze manifest `state/v25_freeze_manifest.json` binds the contract source SHA256, schema version, positive producer fixture and its metadata, schema mutation test source, and JUnit evidence. The frozen loader was not changed after V25QUAL generation.
