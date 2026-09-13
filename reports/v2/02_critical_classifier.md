# V2 Critical Classifier

| model | critical_recall | auprc | brier | ece | review_rate | threshold |
| --- | --- | --- | --- | --- | --- | --- |
| learned_fusion | 1.0000 | 0.9612 | 0.0222 | 0.0136 | 0.2653 | 0.2000 |
| late_probability_fusion | 0.9847 | 0.8549 | 0.0610 | 0.0595 | 0.3400 | 0.2000 |
| focal_loss_pytorch_mlp_cuda | 0.9847 | 0.7646 | 0.0840 | 0.0476 | 0.4312 | 0.2083 |
| weighted_logistic | 0.9847 | 0.7946 | 0.0799 | 0.0284 | 0.4461 | 0.0833 |
| text_linear | 0.9771 | 0.6285 | 0.1129 | 0.0462 | 0.8607 | 0.0614 |
| weighted_xgboost_cuda | 1.0000 | 0.8081 | 0.0770 | 0.0336 | 1.0000 | 0.0000 |

The deployed development candidate uses the weighted logistic route for robustness. Round-4 critical recall was 0.9697 at a 0.5802 critical-positive prediction rate. The latter artifact field is mislabeled as critical_review_coverage; it is not actual review coverage. Sources: `results/v2/critical_classifier.parquet` and `results/v2/selected_candidate_internal_eligibility_r4.parquet`.
