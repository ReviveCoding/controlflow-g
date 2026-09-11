# P24 Independent Review — Repeat 3 Dispositions

Snapshot reviewed: `b1aee1457cc72ffb52bd5299762da8da6017f8be`.

The locked test was not accessed. All seven material findings below were classified `VALID`; P25 remained blocked while repairs and validation-only reruns were performed.

| Finding | Class | Disposition and repair |
|---|---|---|
| Holdout lacked frozen PIT transformation | BLOCKER | `VALID`: one reusable bitemporal transform now serves development and P26; Parquet schema metadata is checked before seal consumption and row data is transformed only after authorized access. |
| CFR staging corpus absent from freeze | BLOCKER | `VALID`: exact CFR Parquet is frozen and recursively hash-verified; mutation regression added. |
| LLM context bypassed typed tools and exact evidence was injected | BLOCKER | `VALID`: exact-ID and synthetic-policy completion removed. LLM-visible retrieval, risk, and anomaly observations now originate inside authorized typed-tool implementations, and retrieval misses remain misses. |
| CFR trust was not freeze-bound | BLOCKER | `VALID`: staging artifacts are hash-checked against the artifact manifest and supplied frames must equal the verified staged frames before evidence receives trusted-ingestion status. P25 also freezes both corpora. |
| Security verdicts were component-only for several attacks | HIGH | `VALID`: every S01–S15 row now requires both its causal component invariant and an integrated workflow invariant; S10, S11, S13, S14, and S15 execute their named boundaries. Matched S14 control performs a real signed approval and action. |
| Orphan action rows evaded audit verification | HIGH | `VALID`: action/event anti-join and legal transition validation added; orphan and mutation regressions added. |
| Tool/authorization audit evidence was ephemeral and co-located | HIGH | `VALID`: registry events persist to a separate hash chain; trust key and signed anchors live under `state/audit_trust`, outside mutable ledger directories; tamper regression added. |

Post-repair nonsealed checks at the time of this disposition:

- Ruff: pass.
- strict mypy: pass for 70 source files.
- pytest: 45 passed.
- integrated S01–S15 run: 15/15 blocked with causal invariants satisfied; zero matched-control false-positive blocks.
- repaired AG6 validation: nominal task success 0.1375, Safe Task Completion 0.1000, evidence coverage 0.1250, temporal correctness 1.0000, P95 latency 4.012343 seconds (`n=80`). The lower STC is retained as the honest result after removing oracle-like evidence completion.

Repeat-4 independent review is required on the resulting clean commit before P24 can complete.
