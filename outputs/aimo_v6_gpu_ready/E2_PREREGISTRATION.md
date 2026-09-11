# E2 preregistration: fixed multi-layer vote

Frozen before E2 execution on 2026-09-10 (Asia/Shanghai).

## Motivation

E1 passed three of four numerical gates, but selected eight different layers
across 25 outer fits and failed one source-transfer direction. Twenty of 25
outer fits selected `C=0.001`. E2 tests whether removing layer selection and
using the consistently preferred regularization reduces variance.

## Fixed method

- Input: the already extracted last-prompt-token representations; no new GPU run.
- Layers: `[4, 8, 12, 16, 20, 24, 28, 32, 36]`.
- One balanced L2 logistic probe per layer.
- Regularization: fixed `C=0.001`; no inner hyperparameter search.
- Each probe is standardized using its training partition only.
- Within an outer fold, the nine binary probe decisions receive equal weight.
- Across seeds, all 45 decisions receive equal weight; final threshold is 50%.
- Outer evaluation: five problem-grouped folds for seeds 42–46.
- Source stress tests: both train-source → other-source directions.
- Negative control: the identical procedure after one deterministic label
  permutation (seed 20260910), evaluated with seed 42.
- Uncertainty: 5,000 problem-level bootstrap samples (seed 20260910).

## Unchanged numerical gates

- grouped OOF balanced accuracy ≥ 0.65;
- bootstrap 95% lower bound > 0.55;
- both source-transfer balanced accuracies ≥ 0.60;
- advantage over shuffled-label control ≥ 0.05.

## Interpretation boundary

The E1 source-transfer results have already been observed. Therefore E2 is a
sequential exploratory experiment, not a new untouched confirmation. A PASS
would justify artifact engineering and one independent Codabench confirmation;
it would not retroactively make the public source holdouts blind.
