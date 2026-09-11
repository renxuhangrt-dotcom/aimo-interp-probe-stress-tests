# E2 fixed multi-layer vote

Status: **PASS** (sequential exploratory validation)

- Layers: [4, 8, 12, 16, 20, 24, 28, 32, 36]
- Fixed C: 0.001
- Grouped OOF balanced accuracy: 0.704758
- Ordinary accuracy: 0.656934
- Bootstrap 95% lower bound: 0.616944
- Randomized-label control: 0.585809
- Advantage over randomized control: 0.118949
- Source holdouts: `{"hendrycks-math-test-reference-100 -> hendrycks-math-train-reference-500": 0.675, "hendrycks-math-train-reference-500 -> hendrycks-math-test-reference-100": 0.7073170731707317}`
- Predicted positives: 69 / 137

## Gate failures

- None

The source-transfer data were observed during E1, so even a PASS here is not an
untouched confirmation. No artifact is exported by this script.
