# E8 final problem-token representation vote

Status: **FAIL** (sequential exploratory validation)

- Representation: `problem_last`
- Layers: [4, 8, 12, 16, 20, 24, 28, 32, 36]
- Fixed C: 0.001
- Grouped OOF balanced accuracy: 0.694857
- V6 grouped OOF balanced accuracy: 0.704758
- Difference versus V6: -0.009901
- Candidate bootstrap 95% lower: 0.610076
- Paired bootstrap difference 95% interval: [-0.07138038619579104, -0.009615384615384581, 0.050605060506050514]
- Source holdouts: `{"hendrycks-math-test-reference-100 -> hendrycks-math-train-reference-500": 0.6583333333333333, "hendrycks-math-train-reference-500 -> hendrycks-math-test-reference-100": 0.6618625277161863}`
- Mean source holdout: 0.660098
- Randomized-label advantage: 0.117024
- Disagreement versus V6: 0.072993
- Corrected/broken versus V6: 4 / 6
- Predicted positives: 71 / 137

## Gate failures

- balanced-accuracy improvement -0.009901 < 0.02
- bootstrap lower 0.610076 is not > V6 0.616944
- paired bootstrap difference lower -0.071380 is not > 0
- mean source holdout 0.660098 < 0.69
- OOF disagreement versus V6 0.072993 < 0.08

No artifact is exported by this validation script.
