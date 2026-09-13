# V2 Fusion Study

| model | critical_recall | review_rate | auprc | brier | ece |
| --- | --- | --- | --- | --- | --- |
| text_embedding_linear | 1.0000 | 1.0000 | 0.5516 | 0.1332 | 0.0964 |
| learned_mlp_fusion | 0.9313 | 0.2421 | 0.9087 | 0.0471 | 0.0541 |
| text_embedding_xgboost_cuda | 0.9618 | 0.3085 | 0.8643 | 0.0628 | 0.0489 |
| late_embedding_tabular_fusion | 0.9389 | 0.2819 | 0.8741 | 0.0531 | 0.0243 |
| tabular_xgboost_cuda | 0.9618 | 0.4345 | 0.8081 | 0.0770 | 0.0336 |

Severity/root fusion results:

| model | accuracy | macro_f1 | critical_recall | critical_ece | review_coverage |
| --- | --- | --- | --- | --- | --- |
| flat_text_tabular | 0.8375 | 0.8198 | 0.9542 | 0.0633 | 0.2405 |
| hierarchical_text_tabular | 0.8458 | 0.8352 | 1.0000 | 0.0752 | 0.2687 |
| hierarchical_selected_calibrated_critical | 0.8706 | 0.8564 | 0.9237 | 0.0136 | 0.2106 |

Sources: `results/v2/fusion.parquet` and `results/v2/hierarchical_severity.parquet`.
