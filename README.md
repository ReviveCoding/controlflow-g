# ControlFlow-G

ControlFlow-G evaluates a tool-using LLM agent for investigating synthetic financial-control exceptions. The system separates evidence retrieval and model-generated explanations from authorization, human-review requirements, and state-changing operations.

This is a local research simulation. It does not connect to banking systems or perform real financial actions.

## Architecture

A case moves through these stages:

1. Retrieve case data and the policy version applicable at the decision time.
2. Estimate risk and generate a bounded, structured investigation.
3. Verify evidence and enforce policy independently of the LLM.
4. Require approval where applicable and commit simulated actions transactionally.
5. Record audit events and evaluate the entire workflow.

The LLM cannot authorize its own actions or alter evaluator truth. Read-only queries and typed actions enforce the interface boundary; policy is rechecked before simulated commits.

For component details, see [the project specification](PROJECT_SPEC.md) and [technical report](TECHNICAL_REPORT_V26.md).

## Frozen V2.6 results

The final experiment used one pinned Qwen3/vLLM configuration and 600 sealed synthetic cases, with observed maximum LLM concurrency of two.

| Metric | Frozen final result |
| --- | ---: |
| Safe Task Completion | 489/600 (81.50%) |
| Critical recall | 141/144 (97.92%) |
| Stale-policy errors | 0/150 |
| Temporal policy accuracy | 600/600 |
| Evidence requirements satisfied | 1,005/1,005 |
| Structured-output failures | 0/600 |
| Unauthorized commits | 0/83 |
| Approval-bypass commits | 0/255 |
| Duplicate commits | 0/600 |
| End-to-end P95 latency | 7.392 seconds |
| Predefined research-release gates | 17/17 passed |

Safe Task Completion requires the task result, evidence, policy timing, and authorization to be correct, with no critical policy violation. It is deliberately stricter than nominal task success.

The zero-event rows apply only to the tested opportunity sets. They do not establish zero risk in a live deployment or generalize to other institutions, models, or hardware.

Primary records: [release decision](state/v26_release_decision.json), [gate decisions](results/v26/final/gate_decision.json), and [denominators](results/v26/final/denominators.json).

## Engineering and evaluation decisions

- **Independent authority:** the model can propose and explain; deterministic code verifies evidence, policy, approval, and commits.
- **Time-valid retrieval:** evaluation checks whether the agent selected the policy that applied when a case occurred, not a later revision.
- **Idempotent execution:** a transactional SQLite ledger prevents repeated simulated side effects and records outcomes.
- **Failure-driven development:** earlier runs uncovered evaluation-truth leakage, denominator errors, mutable SQLite state, and checkpoint-schema issues. Those runs were retained as diagnostic history.
- **Fresh confirmation:** structural corrections were followed by a new qualification, independent review, and a separate one-shot final evaluation.

These decisions and their limitations are documented in [the V2.6 report](TECHNICAL_REPORT_V26.md).

## Reproducing software and evidence checks

The immutable research result belongs to the annotated [controlflow-g-v26-promote tag](https://github.com/ReviveCoding/controlflow-g/tree/controlflow-g-v26-promote). The public package has a separate [controlflow-g-v26-public tag](https://github.com/ReviveCoding/controlflow-g/tree/controlflow-g-v26-public). Neither should be rewritten or treated as a fresh evaluation.

Portable verification uses Python 3.11 and the locked environment:

    uv sync --locked --extra dev
    uv run ruff format --check src tests
    uv run ruff check src tests
    uv run mypy src
    uv run pytest -q -m "not gpu and not spark and not network and not sealed"
    uv run python scripts/verify_release_portability.py

The [reproducibility guide](REPRODUCIBILITY.md) explains artifact hashes, historical Windows line endings, and why the frozen cases must not be rerun for model selection. GitHub Actions runs the portable checks.

## Repository guide

| Directory | Contents |
| --- | --- |
| src/controlflow/ | Agent runtime, models, retrieval, authorization, audit, and evaluation |
| configs/ | Resource limits, schemas, policy versions, and release criteria |
| tests/ | Unit, integration, and failure-path checks |
| data/, artifacts/, results/ | Synthetic datasets and measured evidence |
| state/ | Frozen manifests, receipts, and release decisions |
| reports/ | Evaluation and technical reports |

## Limits and licensing

The benchmark uses public-source grounding and synthetic enterprise truth on a local RTX 4090 Laptop GPU. It does not measure analyst time saved, real human-review quality, bank production safety, or operating costs at customer scale.

ControlFlow-G is distributed under the [MIT License](LICENSE). See [security reporting](SECURITY.md) and [reproducibility](REPRODUCIBILITY.md) for implementation and validation boundaries.
