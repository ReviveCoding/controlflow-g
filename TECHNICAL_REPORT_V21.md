# ControlFlow-G V2.1 technical report

## Outcome

V2.1 closed as `V21_DEVELOPMENT_NO_GO` at phase V21-19. No qualification or final holdout was created or consumed. Historical V1/V2 evidence was not modified.

## Verified boundary and environment

The active branch was `v2.1-development`. The annotated `controlflow-g-v2-no-go` tag peeled to `2d21a70c1585669a456929c554bdbfc997f708ac` and was an ancestor of HEAD. Installed WSL evidence showed vLLM 0.29.0, PyTorch 2.13.0+cu130, CUDA 13.0, and an RTX 4090 Laptop GPU (`state/v21_environment_manifest.json`). The explicit xgrammar configuration was accepted, but server startup failed for insufficient KV-cache memory while another workload occupied the GPU (`state/v21_vllm.stderr.log`).

## Engineering evidence

The full local test suite passed 116 tests (`build/v21-pytest.xml`), including 21 V2.1-focused tests. These establish unit behavior only. Development metrics in `results/v21/development_experiments.parquet` and paired intervention outputs are marked diagnostic-only after reviewers demonstrated circular truth derivatives and proxy measurements.

## Independent review and decision

Correctness, methodology, and security/reliability reviewers all denied clearance. The main agent classified 14 blockers and 12 high findings as valid in `state/v21_review_findings.json`. Integrity gating therefore prohibits qualification (`state/v21_integrity.json`). Release decision: `NO_PROMOTE`.
