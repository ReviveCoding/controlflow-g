# Development and Evaluation Guidelines

ControlFlow-G is a research implementation of a governed tool-using agent. The frozen V2.6 results and their associated evaluation identities must remain reproducible.

## Code changes

- Keep implementation in src/controlflow/ with typed interfaces, explicit state changes, and tests.
- Treat retrieved text as untrusted data. The LLM must not grant authorization, change evaluator truth, or execute unbounded writes.
- Restrict SQL to approved read-only operations. Use typed, idempotent interfaces for simulated state changes.
- Preserve temporal policy checks, approval requirements, durable execution state, and the audit trail.
- Make no claims about production deployment or real financial actions.

## Evaluation integrity

- Keep training, development, qualification, final, adversarial, and synthetic-production evidence distinct.
- Do not alter frozen holdouts, decision rules, historical results, or the promoted research tag.
- Record observed device, configuration, timing, and experiment identity when running new experiments.
- Do not infer zero risk from zero observed failures in a limited sample.
- Treat failed or invalidated runs as recorded evidence; use a new identity after a material evaluation change.

## Review and validation

- Review correctness, methodology, and security separately for changes that affect scientific claims or execution boundaries.
- Reproduce findings with file references, tests, or evidence artifacts. Reviewers do not edit the artifacts they assess.
- Before merging changes, run the portable checks in README.md and verify the immutable research release through REPRODUCIBILITY.md.

The historical study protocol is in PROJECT_SPEC.md. Any new scientific study requires a new protocol and evaluation identity.
