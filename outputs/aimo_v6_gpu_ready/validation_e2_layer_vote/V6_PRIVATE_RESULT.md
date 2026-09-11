# V6 private Small Track result

- Accuracy: `0.631578947368421` (12/19)
- Coverage: `1.0`
- Invalid predictions: `0`
- Predicted robust: 11
- Predicted non-robust: 8
- Derived confusion matrix: TP=7, TN=5, FP=4, FN=3
- Derived balanced accuracy: approximately `0.627778`

Relative to prior submissions, V6 gained two correct cases over V4/V5 and
three over the all-negative/probe-control baseline. This is the first submitted
representation method in the project to demonstrate a non-constant private-set
improvement.

The private set contains only 19 cases. V6 is frozen as a checkpoint; its
private score and predictions must not be used for per-case, threshold, or
feature tuning.
