# V2.5 resume evidence

The historical V2.4 boundary is the unchanged `controlflow-g-v24-no-go` tag at `b3ee374131cfe20985cdd685b7bb1ee210bb22e4`. V2.5 executable source and the successful 60-case development rehearsal are committed at `3bc7194`. The reviewed prequalification freeze is committed at `578cba3` with freeze hash `255bf24722b2e8774f0cce8d12dbd4e300f09a2aa92295570ba87d7ca6f1f294`.

Exactly one V25QUAL cohort was generated. Its structural admission passed, but the frozen pre-candidate binding check failed. `state/v25_qualification_manifest.json` is `V25_DEVELOPMENT_NO_GO`; `state/v25_qualification_failure.json` records the cause. There was no candidate execution and no final holdout. V25QUAL must never be reused or rerun. The recorded release decision is `NO_PROMOTE`.

For future continuation, start a new version and holdout identity after repairing and reviewing the stable contamination comparison in development. See `artifacts/v25_resume_evidence.json` for machine-readable pointers.
