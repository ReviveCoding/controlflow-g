# V2 Model Tournament

Development-only screening results:

| architecture | cases | structured_output_failure_rate | critical_recall | safe_task_completion | p95_latency_seconds |
| --- | --- | --- | --- | --- | --- |
| V2-A0_v1_era_0.5b_unconstrained | 32 | 1.0000 | 0.0000 | 0.0000 | 1.8550 |
| V2-A1_qwen3_unconstrained | 32 | 1.0000 | 0.0000 | 0.0000 | 11.9756 |
| V2-A2_qwen3_schema | 32 | 0.0000 | 0.7143 | 0.2500 | 15.2549 |
| V2-A3_decomposed_flat | 32 | 0.0000 | 0.8571 | 0.8438 | 17.9439 |
| V2-A4_decomposed_hierarchical | 32 | 0.0000 | 1.0000 | 0.8438 | 17.3575 |

The 32-case screen is architecture screening only; it is not eligibility or final evidence. Source: `results/v2/model_tournament_all.parquet`.
