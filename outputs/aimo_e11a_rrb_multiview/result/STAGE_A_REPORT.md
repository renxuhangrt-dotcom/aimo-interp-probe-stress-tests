# E11a Stage A result

Status: **FAIL**

## Metrics

```json
{
  "candidate_balanced_accuracy": 0.6273377337733774,
  "candidate_ordinary_accuracy": 0.635036496350365,
  "candidate_bootstrap_95_lower": 0.5344018332857619,
  "v6_frozen_oof_balanced_accuracy": 0.7047579757975797,
  "paired_improvement_bootstrap_95_lower": -0.1562518210955711,
  "source_holdout_balanced_accuracy": {
    "hendrycks-math-test-reference-100 -> hendrycks-math-train-reference-500": 0.59,
    "hendrycks-math-train-reference-500 -> hendrycks-math-test-reference-100": 0.7172949002217295
  },
  "mean_source_holdout_balanced_accuracy": 0.6536474501108647,
  "randomized_control_balanced_accuracy": 0.5936468646864687,
  "randomized_control_advantage": 0.03369086908690877,
  "disagreement_vs_v6": 0.18248175182481752,
  "predicted_positive": 58,
  "actual_positive": 36,
  "confusion_tn_fp_fn_tp": [
    65,
    36,
    14,
    22
  ]
}
```

## Gate failures

- candidate BA 0.627338 < 0.724758
- candidate bootstrap lower 0.534402 <= 0.616944
- paired bootstrap improvement lower -0.156252 <= 0
- source holdout hendrycks-math-test-reference-100 -> hendrycks-math-train-reference-500 0.590000 < 0.65
- mean source holdout 0.653647 < 0.69
- random-control advantage 0.033691 < 0.10
