# Security policy

ControlFlow-G is a synthetic research prototype. It is not a bank deployment and cannot perform real financial actions.

The public package is distributed under the MIT License in [LICENSE](LICENSE). Licensing does not turn the research prototype into a production security claim.

Report suspected vulnerabilities through GitHub's private vulnerability reporting mechanism if enabled for this repository. Otherwise, open a GitHub issue containing only non-sensitive coordination details and request a private channel. Do not post exploit details, credentials, private keys, or sensitive logs publicly before coordination.

The intended boundaries are read-only parsed SQL, typed and allowlisted simulated actions, independent PDP/PEP authorization, bound approvals, evidence verification, idempotent SQLite execution, and a tamper-evident local audit chain. The local chain depends on a trusted head anchor. Normal portable CI requires no real secrets. Local private signing keys are intentionally excluded from Git; public key material and synthetic evidence may be committed.
