# E7 problem-token mean representation vote

Status: **FAIL** (sequential exploratory validation)

- Representation: `problem_mean`
- Layers: [4, 8, 12, 16, 20, 24, 28, 32, 36]
- Fixed C: 0.001
- Grouped OOF balanced accuracy: 0.718647
- V6 grouped OOF balanced accuracy: 0.704758
- Difference versus V6: 0.013889
- Candidate bootstrap 95% lower: 0.639071
- Paired bootstrap difference 95% interval: [-0.06757295573085055, 0.013054614558373867, 0.09400428503759306]
- Source holdouts: `{"hendrycks-math-test-reference-100 -> hendrycks-math-train-reference-500": 0.6116666666666667, "hendrycks-math-train-reference-500 -> hendrycks-math-test-reference-100": 0.7560975609756098}`
- Mean source holdout: 0.683882
- Randomized-label advantage: 0.198295
- Disagreement versus V6: 0.124088
- Corrected/broken versus V6: 9 / 8
- Predicted positives: 70 / 137

## Gate failures

- balanced-accuracy improvement 0.013889 < 0.02
- paired bootstrap difference lower -0.067573 is not > 0
- source holdout hendrycks-math-test-reference-100 -> hendrycks-math-train-reference-500 0.611667 < 0.65
- mean source holdout 0.683882 < 0.69

No artifact is exported by this validation script.
