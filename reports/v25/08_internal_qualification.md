# Internal qualification

One V25QUAL cohort was generated with seed 25051 after freeze. `results/v25/qualification/structural_admission.json` records 600 unique cases, 146 critical, 150 stale-policy challenges, 145 certified review opportunities, and 69 certified deny opportunities. `results/v25/qualification/contamination.json` records zero findings across 87 prior runtime cohorts.

The frozen runner stopped in its pre-candidate generation-binding check with `V25_HOLDOUT_GENERATION_BINDING_INVALID`. The stored dataset-manifest and contamination-report hashes match their files. The report-object equality check includes a newly generated `created_at` timestamp, so it fails on recomputation. No candidate checkpoint or output file was created. `state/v25_qualification_failure.json` records the failure; the one-shot manifest is `V25_DEVELOPMENT_NO_GO`. The 600-case structural counts are not qualification performance evidence.
