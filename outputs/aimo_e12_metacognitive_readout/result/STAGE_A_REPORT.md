# E12 Stage A result

Status: **FAIL**

## Metrics

```json
{
  "candidate_balanced_accuracy": 0.6362761276127613,
  "candidate_ordinary_accuracy": 0.635036496350365,
  "candidate_bootstrap_95_lower": 0.540968263962531,
  "v6_frozen_oof_balanced_accuracy": 0.7047579757975797,
  "paired_improvement_bootstrap_95_lower": -0.17426179846938777,
  "source_holdout_balanced_accuracy": {
    "hendrycks-math-test-reference-100 -> hendrycks-math-train-reference-500": 0.53,
    "hendrycks-math-train-reference-500 -> hendrycks-math-test-reference-100": 0.5565410199556541
  },
  "mean_source_holdout_balanced_accuracy": 0.543270509977827,
  "randomized_control_balanced_accuracy": 0.6303630363036303,
  "randomized_control_advantage": 0.00591309130913098,
  "disagreement_vs_v6": 0.27007299270072993,
  "predicted_positive": 60,
  "actual_positive": 36,
  "confusion_tn_fp_fn_tp": [
    64,
    37,
    13,
    23
  ],
  "option_order_diagnostics": {
    "robustness": {
      "mean_absolute_margin_difference": 2.759150266647339,
      "semantic_sign_agreement": 0.2871046228710462,
      "pearson_correlation": 0.11756741977941283
    },
    "correctness": {
      "mean_absolute_margin_difference": 3.7866451740264893,
      "semantic_sign_agreement": 0.08353609083536091,
      "pearson_correlation": -0.7706016937500715
    }
  }
}
```

## Gate failures

- candidate BA 0.636276 < 0.724758
- candidate bootstrap lower 0.540968 <= 0.616944
- paired bootstrap improvement lower -0.174262 <= 0
- source holdout hendrycks-math-test-reference-100 -> hendrycks-math-train-reference-500 0.530000 < 0.65
- source holdout hendrycks-math-train-reference-500 -> hendrycks-math-test-reference-100 0.556541 < 0.65
- mean source holdout 0.543271 < 0.69
- random-control advantage 0.005913 < 0.10
