# Stratified generator

The V2.4 generator deterministically selects required strata before a frozen-seed permutation (`src/controlflow/v24/generator.py`, `configs/v24/strata.yaml`). Three development-only seeds were admitted (`results/v24/development_generator_validation.json`). The one V24QUAL generation produced 600 unique cases and passed pre-execution structural admission (`results/v24/qualification/structural_admission.json`); contamination findings were zero (`results/v24/qualification/contamination.json`). No replacement V24QUAL was generated.
