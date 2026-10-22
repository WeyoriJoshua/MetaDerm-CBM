# Mapping code outputs to manuscript evidence

The paths below assume the default `efficientnet_b3` experiment. The same structure is created under `results/swin_tiny/` when the Swin override is used.

| Manuscript component | Primary output |
|---|---|
| Dataset/split table | `results/efficientnet_b3/audit/split_summary.csv` |
| Concept-distribution appendix | `results/efficientnet_b3/audit/concept_summary.csv` |
| Concept inconsistency analysis | `results/efficientnet_b3/audit/training_profile_reliability.csv` |
| Concept model performance | `results/efficientnet_b3/final/seed*/concept_metrics.csv` |
| Main diagnostic table | `results/efficientnet_b3/final/seed*/metrics.json` + CBM metrics |
| Calibration claims | Brier/ECE in per-seed `metrics.json` |
| Selective prediction | `results/efficientnet_b3/final/seed*/risk_coverage.csv` + figure |
| Metacognitive action rates | `results/efficientnet_b3/final/seed*/action_summary.csv` |
| Neural-symbolic disagreement | `results/efficientnet_b3/final/seed*/test_predictions.csv` |
| Intervention analysis | `results/efficientnet_b3/intervention/seed*/one_step_oracle_intervention.csv` |
| Component contribution | `results/efficientnet_b3/ablation/seed*/ablation_summary.csv` |
| Confidence intervals | `results/efficientnet_b3/final/seed*/bootstrap_statistics.json` |
| Representation analysis | `results/efficientnet_b3/representation/seed*/test/` |
| Concept-noise robustness | `results/efficientnet_b3/robustness/seed*/concept_noise_robustness.csv` |
| Post-hoc XAI baseline | `results/efficientnet_b3/gradcam/seed*/test/` |
| Aggregated LaTeX/CSV tables | `results/efficientnet_b3/publication/tables/` |

Every manuscript number should be traceable to one of these machine-readable outputs. Avoid manually transcribing results when a generated table can be used instead.
