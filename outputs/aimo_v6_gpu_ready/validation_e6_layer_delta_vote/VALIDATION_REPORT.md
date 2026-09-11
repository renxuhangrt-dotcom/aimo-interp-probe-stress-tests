# E6 adjacent-layer representation-change vote

Status: **FAIL** (sequential exploratory validation)

- Deltas: ['4-0', '8-4', '12-8', '16-12', '20-16', '24-20', '28-24', '32-28', '36-32']
- Fixed C: 0.001
- Grouped OOF balanced accuracy: 0.713696
- V6 grouped OOF balanced accuracy: 0.704758
- Difference versus V6: 0.008938
- Candidate bootstrap 95% lower: 0.630798
- Paired bootstrap difference 95% interval: [-0.014150943396226356, 0.007345360824742242, 0.042868431855500724]
- Source holdouts: `{"hendrycks-math-test-reference-100 -> hendrycks-math-train-reference-500": 0.675, "hendrycks-math-train-reference-500 -> hendrycks-math-test-reference-100": 0.6618625277161863}`
- Mean source holdout: 0.668431
- Randomized-label advantage: 0.150715
- Disagreement versus V6: 0.014599
- Corrected/broken versus V6: 1 / 1
- Predicted positives: 71 / 137

## Gate failures

- balanced-accuracy improvement 0.008938 < 0.02
- paired bootstrap difference lower -0.014151 is not > 0
- mean source holdout 0.668431 < 0.69
- OOF disagreement versus V6 0.014599 < 0.08

No artifact is exported by this validation script.
