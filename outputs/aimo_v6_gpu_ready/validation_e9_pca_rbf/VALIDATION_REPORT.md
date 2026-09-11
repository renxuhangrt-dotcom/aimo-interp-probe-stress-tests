# E9 PCA-RBF layer vote

Status: **FAIL** (sequential exploratory validation)

- Representation: `input_last_token`
- Layers: [4, 8, 12, 16, 20, 24, 28, 32, 36]
- PCA components: 10
- RBF SVC: C=1.0, gamma=`scale`, class weight=`balanced`
- Grouped OOF balanced accuracy: 0.712596
- V6 grouped OOF balanced accuracy: 0.704758
- Difference versus V6: 0.007838
- Candidate bootstrap 95% lower: 0.623326
- Paired bootstrap difference 95% interval: [-0.049604382227632415, 0.008843913237477574, 0.06074766355140182]
- Source holdouts: `{"hendrycks-math-test-reference-100 -> hendrycks-math-train-reference-500": 0.6283333333333333, "hendrycks-math-train-reference-500 -> hendrycks-math-test-reference-100": 0.7682926829268293}`
- Mean source holdout: 0.698313
- Randomized-label advantage: 0.121012
- Disagreement versus V6: 0.109489
- Corrected/broken versus V6: 11 / 4
- Predicted positives: 56 / 137

## Gate failures

- balanced-accuracy improvement 0.007838 < 0.02
- paired bootstrap difference lower -0.049604 is not > 0
- source holdout hendrycks-math-test-reference-100 -> hendrycks-math-train-reference-500 0.628333 < 0.65

No artifact is exported by this validation script.
