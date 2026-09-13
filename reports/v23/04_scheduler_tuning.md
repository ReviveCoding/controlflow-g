# V2.3 Scheduler Tuning

The tournament records balanced/interactivity and 1024/2048/4096 batched-token experiments. Selection uses sustained end-to-end P95, structured validity, and unchanged safety results, not throughput alone.

- `ablation_interactivity_current_schema_160_b2048_warm_60`: n=60, total P95=23.700s, structured failure=0.0000, Core STC=0.8167.
- `ablation_interactivity_minimal_128_installed_default_batch_warm_60`: n=60, total P95=5.485s, structured failure=0.0000, Core STC=0.8167.
- `ablation_interactivity_minimal_enum_160_b2048_warm_60`: n=60, total P95=8.933s, structured failure=0.0000, Core STC=0.8167.
- `ablation_selected_interactivity_minimal_128_b2048_cold_60`: n=60, total P95=6.181s, structured failure=0.0000, Core STC=0.8167.
- `baseline_v22_cold_60_r2`: n=60, total P95=17.027s, structured failure=0.0000, Core STC=0.8167.
- `p0_balanced_current_160_b2048_warm_60`: n=60, total P95=17.815s, structured failure=0.0000, Core STC=0.8167.
- `p0_balanced_minimal_128_b2048_warm_60`: n=60, total P95=10.619s, structured failure=0.0000, Core STC=0.8167.
- `p0_balanced_minimal_enum_128_b2048_warm_60`: n=60, total P95=5.737s, structured failure=0.0000, Core STC=0.8167.
- `p0_balanced_minimal_enum_96_b2048_warm_60`: n=60, total P95=7.773s, structured failure=0.0000, Core STC=0.8167.
- `p0_balanced_minimal_exact_128_b2048_warm_60`: n=60, total P95=10.008s, structured failure=1.0000, Core STC=0.8167.
- `p1_interactivity_minimal_enum_128_b1024_warm_60`: n=60, total P95=6.825s, structured failure=0.0000, Core STC=0.8167.
- `p1_interactivity_minimal_enum_128_b2048_warm_60`: n=60, total P95=5.826s, structured failure=0.0000, Core STC=0.8167.
- `p1_interactivity_minimal_enum_128_b4096_warm_60`: n=60, total P95=7.044s, structured failure=0.0000, Core STC=0.8167.
- `selected_balanced_minimal_enum_128_b2048_cold_60`: n=60, total P95=7.076s, structured failure=0.0000, Core STC=0.8167.
- `stage_b_balanced_minimal_enum_128_b2048_warm_400`: n=400, total P95=8.336s, structured failure=0.0000, Core STC=0.8100.
- `stage_b_interactivity_minimal_enum_128_b2048_warm_400_r2`: n=400, total P95=8.022s, structured failure=0.0000, Core STC=0.8100.
- `stage_c_selected_interactivity_minimal_128_b2048_r22_warm_600_independent`: n=600, total P95=6.169s, structured failure=0.0000, Core STC=0.8433.
- `stage_c_selected_interactivity_minimal_128_b2048_r23_warm_600_recovery_r3`: n=600, total P95=6.448s, structured failure=0.0000, Core STC=0.8133.
