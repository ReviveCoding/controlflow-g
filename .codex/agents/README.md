# Independent Reviewers

These roles are independent, read-only assurance reviewers. They inspect the same named snapshot in parallel and do not read one another's findings before submission.

- Never edit, create, delete, format, or regenerate repository files; never repair findings.
- Before P26, never inspect locked-test records, labels, payloads, predictions, or metrics. Opaque split hashes and sealing logic may be inspected.
- Cite reproducible evidence: file and line, artifact/hash, or exact diagnostic command and relevant output. Missing evidence is an assurance gap, not a pass.
- Report negative and null findings to the main agent. Findings begin as `PROPOSED`; the main agent classifies them `VALID`, `INVALID`, or `NOT_APPLICABLE` and owns repairs.

Each finding includes ID, severity (`BLOCKER|HIGH|MEDIUM|LOW`), category, finding, evidence, impact, remediation, affected phases/artifacts, final-test implications, and confidence. Each review ends with snapshot ID, checks, unavailable evidence, `locked-test-accessed: false`, and `PASS|PASS_WITH_FINDINGS|FAIL`.

Boundaries: Correctness owns implementation/runtime and mechanical traceability; Methodology owns scientific validity and claims; Security owns adversarial boundary enforcement.

