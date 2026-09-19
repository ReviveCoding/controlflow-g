# Development regression

The single new 120-case development regression used the frozen Qwen revision on the local RTX 4090 at concurrency two. Its measured results, including critical recall 31/31, Core STC 95/120, temporal accuracy 120/120, evidence completeness 186/186, zero observed security commits, and total P95 5.398303 seconds, are in `results/v24/development_regression/summary.json`. This was development-only evidence, not V23QUAL-based selection and not proof of zero true risk.
