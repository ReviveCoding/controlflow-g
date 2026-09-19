# V2.4 resume evidence

V24QUAL is permanently consumed diagnostic evidence. Do not regenerate it, resume candidate execution, patch the frozen verifier and rerun it, or generate V24FINAL under this iteration. The terminal decision is `NO_PROMOTE` because the post-close verifier failed; no final gate result exists.

The reviewed executable boundary was committed at `eddbd98`, and the qualification freeze was committed at `f1aa8d3c6acfba0ec120b570712e4266532a4b5d` with freeze hash `6f30549a0c5d74a215eb853b305dad29cf916d64622a9b96d1d5d095319bca81`. The qualification manifest records `V24_DEVELOPMENT_NO_GO` after one 600-case execution. SQLite finalization and stability succeeded, but `state/v24_postclose_verification.json` records `freeze_or_denominator_verification:'critical_threshold'`.

The frozen verifier read `checkpoint["critical_threshold"]`; the actual checkpoint stores this under `checkpoint["fields"]["critical_threshold"]`. This code defect is in the committed, frozen protocol. The denominator report and aggregate table are diagnostic only, not admissible qualification evidence. A new iteration with a new holdout is required for any corrected verifier.
