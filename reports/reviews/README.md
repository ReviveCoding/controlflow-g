# Review Schedule

- P05–P07: correctness + methodology review lakehouse, temporal semantics, point-in-time features, and leakage.
- P09: methodology split/sealing review; correctness enforcement/manifest review.
- P15: methodology retrieval evaluation; security temporal/authorization/poisoning boundaries.
- P18: correctness + security orchestration, tools, authorization, HITL, idempotency.
- P20: correctness + security failure/recovery evidence.
- P23: methodology validation statistics, gates, and selection provenance.
- P24: all three independently and in parallel against one recorded snapshot, without locked-test content.
- P25: freeze completeness/hash attestation; P26: exactly one locked evaluation.

The main agent writes `results/review_findings.jsonl` and `results/review_dispositions.jsonl`, repairs validated material issues, reruns affected validation, and repeats affected reviews. Post-final invalidating defects invoke the new-holdout rule.
