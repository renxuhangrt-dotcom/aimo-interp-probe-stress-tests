# Pre-compute audit and frozen experiment protocol

Date: 2026-09-10 (Asia/Shanghai)

## Frozen upstream inputs

- Official repository: `https://github.com/aimo-interp/baselines.git`
- Audited commit: `7e8839966750059b6b1d247a12ab56552c79b342`
- Public aggregate input: `data/math-robust-agg.csv`
- Evaluation model label: `qwen3-8b:low`
- Hidden-state checkpoint: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`
- Representation: final prompt-token hidden state from every model layer
- System prompt: the pinned official `prompts/solve.txt`

## Dataset audit

The source CSV has 141 rows but only 137 unique `problem_id` values.  Four rows
are exact, label-consistent duplicates: one extra row for `0d430e`, two extra
rows for `94b3fb`, and one extra row for `db39ee`.  After collapse:

- labels: 101 false, 36 true;
- `hendrycks-math-train-reference-500`: 85 problems (60 false, 25 true);
- `hendrycks-math-test-reference-100`: 52 problems (41 false, 11 true).

The platform-independent canonical-content SHA-256 is
`3FC14BFC02F7C1D0D79BB917491C7710F6A798150416B157C6C34205CEC9E067`.
The machine-readable evidence is in `data/dataset_audit.json`.

## Validation risk found

The audited official classification code applies `StratifiedKFold` to rows.
Because duplicate problem rows exist, a duplicate can cross the train/test
boundary.  This package removes exact duplicates and still uses
`StratifiedGroupKFold(groups=problem_id)` for every outer and inner split.
Normalization, layer selection, and regularization selection are fit only on
the relevant training partition.

## Frozen validation design

- candidate layers: 0, 4, 8, 12, 16, 20, 24, 28, 32, 36;
- logistic probe, balanced class weights, L2 regularization;
- candidate `C`: 0.001, 0.01, 0.1, 1, 10;
- outer validation: 5 grouped folds over seeds 42–46;
- inner selection: 4 grouped folds inside each outer-training partition;
- deployment-style OOF margin: mean of the five seed-specific outer margins;
- source shift: both train-source → other-source directions, with independent
  inner selection using only the training source;
- negative control: labels shuffled once with seed 20260910, then the same
  nested pipeline run using seed 42;
- uncertainty: 5,000 problem-level bootstrap samples, seed 20260910.

## Fail-closed decision

No V6 artifact will be built unless all conditions hold simultaneously:

- OOF balanced accuracy ≥ 0.65;
- bootstrap 95% lower bound > 0.55;
- each source-shift holdout balanced accuracy ≥ 0.60;
- OOF score exceeds the shuffled-label control by ≥ 0.05.

This protocol is frozen before extracting the new hidden states.  Any later
change must be logged as a new experiment rather than silently replacing this
one.
