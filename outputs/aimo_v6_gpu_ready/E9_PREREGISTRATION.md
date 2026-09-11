# E9 preregistration: PCA-RBF layer vote

Frozen before E9 execution on 2026-09-10 (Asia/Shanghai).

## Motivation

The official baseline exposes a kernel alternative with optional PCA, while all
Small candidates so far use linear logistic boundaries. E9 tests a fixed,
low-dimensional nonlinear classifier on the proven V6 representation. This
changes the classifier family without changing token extraction or using hidden
leaderboard outcomes.

## Fixed method

- Input: V6's audited `input_last_token` activation cache.
- Layers: `[4, 8, 12, 16, 20, 24, 28, 32, 36]`.
- Within every training partition and layer, fit randomized PCA with exactly 10
  components and the outer seed; transform held-out data using that PCA only.
- Classifier: RBF SVC with fixed `C=1.0`, `gamma="scale"`, and
  `class_weight="balanced"`.
- Nine equal hard binary votes; fixed `0.5` threshold.
- Five problem-grouped folds for seeds `42..46`.
- Both source-holdout directions and the seed-42 shuffled-label control.
- 5,000 problem-level bootstrap samples, seed `20260910`.
- V6/E2 OOF predictions are used only for paired aggregate comparison.
- No layer, PCA dimension, C, gamma, threshold, or ensemble-weight search.
- Hidden Codabench IDs, predictions, and inferred labels are prohibited from
  training and selection.

## Frozen promotion gates

All must pass:

- grouped OOF balanced accuracy >= V6 + `0.02` (>= `0.7247579757975797`);
- candidate bootstrap 95% lower bound > `0.6169437631511935`;
- paired bootstrap 95% lower bound of `(E9 BA - V6 BA)` > `0`;
- each source-holdout BA >= `0.65` and their mean >= `0.69`;
- advantage over shuffled-label control >= `0.10`;
- OOF disagreement versus V6 >= `0.08`.

Failure of any gate records E9 as negative and produces no submission archive.
Passing permits artifact engineering and exact deployment replay, but not
automatic Codabench submission.

