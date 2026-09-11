# E8 preregistration: final problem-token representation

Frozen before the first evaluation of the previously extracted `problem_last`
view on 2026-09-10 (Asia/Shanghai).

## Motivation

E7's mean over all problem tokens contained a real but source-asymmetric signal.
E8 tests the final token inside the mathematical problem text. It excludes the
system prompt and assistant-generation header like E7, while avoiding lexical
averaging across the full problem span. The representation was extracted
label-blind in the same forward pass as E7 and has not yet been evaluated.

## Fixed method

- Representation: `problem_last` only.
- Exact model/cache: the audited E7 Kaggle archive, FP16
  `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` on T4 x2.
- Layers: `[4, 8, 12, 16, 20, 24, 28, 32, 36]`.
- One balanced L2 logistic probe per layer, fixed `C=0.001`.
- Training-partition-only standardization.
- Nine equal hard binary votes; fixed `0.5` threshold.
- Five problem-grouped folds for seeds `42..46`.
- Both source-holdout directions and the seed-42 shuffled-label control.
- 5,000 problem-level bootstrap samples, seed `20260910`.
- V6/E2 OOF predictions are used only for paired aggregate comparison.
- Hidden Codabench IDs, V6/V7 private predictions, and inferred labels are
  prohibited from training, selection, thresholding, or per-case analysis.

## Frozen promotion gates

All must pass:

- grouped OOF balanced accuracy >= V6 + `0.02` (>= `0.7247579757975797`);
- candidate bootstrap 95% lower bound > `0.6169437631511935`;
- paired bootstrap 95% lower bound of `(E8 BA - V6 BA)` > `0`;
- each source-holdout BA >= `0.65` and their mean >= `0.69`;
- advantage over shuffled-label control >= `0.10`;
- OOF disagreement versus V6 >= `0.08`.

Failure of any gate records E8 as negative and produces no submission archive.
Passing permits artifact engineering and exact deployment replay, but not
automatic Codabench submission.

