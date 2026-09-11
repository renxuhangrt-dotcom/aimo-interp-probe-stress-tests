# E7 preregistration: problem-token mean representation

Frozen before E7 GPU extraction and validation on 2026-09-10 (Asia/Shanghai).

## Motivation

The official baseline and V6 use the last token of the rendered input prompt,
which is normally part of the assistant-generation header rather than a token
inside the mathematical problem. E7 tests whether averaging representations
over the exact user-problem token span provides a more semantic and transferable
signal of robustness.

## Label-blind extraction

- Exact model: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`, FP16 on Kaggle T4 x2.
- Exact pinned official baseline commit and system prompt used by V6.
- The tokenizer's offset mapping must cover the complete, verbatim problem-text
  character span inside the rendered chat prompt; missing or truncated spans
  fail closed.
- For every hidden-state layer, average only tokens overlapping that problem
  span. Special tokens, system-prompt tokens, and assistant-header tokens are
  excluded.
- The same forward pass also saves the last problem-token representation as a
  label-blind future view. It is not evaluated or selected in E7.
- Expected coverage: 137/137 unique public problems, 37 layers, 4096 dimensions,
  finite values only, no CPU/disk model offload.

## Fixed E7 method

- Primary view: `problem_mean` only.
- Layers: `[4, 8, 12, 16, 20, 24, 28, 32, 36]`.
- One balanced L2 logistic probe per layer, fixed `C=0.001`.
- Training-partition-only standardization.
- Equal binary votes and fixed `0.5` threshold, identical to V6/E2.
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
- paired bootstrap 95% lower bound of `(E7 BA - V6 BA)` > `0`;
- each source-holdout BA >= `0.65` and their mean >= `0.69`;
- advantage over shuffled-label control >= `0.10`;
- OOF disagreement versus V6 >= `0.08`.

Failure of any gate records E7 as negative and produces no submission archive.
Passing permits artifact engineering and exact deployment replay, but not
automatic Codabench submission.

