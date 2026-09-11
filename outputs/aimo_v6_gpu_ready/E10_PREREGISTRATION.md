# E10 preregistration: independent Small-model OOD audit

Date frozen: 2026-09-10 (Asia/Shanghai)

## Purpose

E10 is an out-of-distribution audit of the already frozen V6 submission. It is
not a training run, a threshold search, or a candidate-selection exercise. No
V8 archive may be produced from E10 alone.

## Frozen data

- Repository: `aimo-interp/aimo-interp-challenge-sample-full`
- Revision: `c0ffb7e678294cc8819d50d979ba28ba49bd02d0`
- Parquet SHA-256:
  `2989AB6A018F2779A821209F414BD0049C9EF1C2488F589FF2D3BF4ED16942FF`
- Expected complete dataset: 558 rows, ten unique AIMO problems, eight model
  IDs.
- Direct Small slice: rows whose `model_id` is exactly `qwen3-8b:low`.
- One evaluation row is formed per `(dataset_id, problem_id)` pair.
- Frozen label rule:
  `model_is_robust := max(absolute_accuracy_decay over available perturbation rows) <= 0`.

Before any E10 prediction is inspected, the notebook must prove that the
frozen rule reproduces `model_is_robust` for all 137 unique problems in the
official baseline's `math-robust-final.csv`. It must also prove zero overlap
with the 137 training problems by both `problem_id` and normalized original
problem text.

## Frozen representation and predictor

- Checkpoint: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`.
- Prompt: the solve prompt from official baseline commit
  `7e8839966750059b6b1d247a12ab56552c79b342`.
- Representation: final input-token state, all 37 hidden-state outputs, exact
  FP16 weights on Kaggle T4 x2.
- Predictor: the existing schema-v3 V6 artifact with SHA-256
  `7A4DF77D031530DC4998CBC4B6FFF2B8B717F99C7FA04E2646847C6A5BE3672B`.
- Layers and voting remain exactly V6: layers
  `[4, 8, 12, 16, 20, 24, 28, 32, 36]`, 25 grouped probe ensembles, 225 equal
  binary votes, majority threshold 0.5.
- Labels are written into audit metadata but are not supplied to the model
  forward pass or predictor.

## Frozen analysis

After the Kaggle archive is returned, report:

- exact direct-Small sample size and class counts;
- V6 ordinary accuracy (primary, matching the competition metric);
- balanced accuracy when both classes are present;
- always-false and always-true accuracy;
- confusion counts, positive-prediction rate, and a 95% Wilson interval for
  ordinary accuracy;
- V6 vote fractions for diagnosis, without case-specific tuning.

The result is decision-grade only if there are at least eight direct Small
problems and both classes are present. In that case, E10 supports continued
Small-method research only if V6 accuracy is at least 0.60 and exceeds the
better constant baseline by at least 0.10. Otherwise it is a failed OOD audit.
With fewer than eight problems or only one class, the result is explicitly
inconclusive and cannot authorize V8.

Regardless of the outcome, E10 labels, problem IDs, and individual errors may
not be used to tune a threshold, select layers, construct text rules, or alter
individual predictions. The private 19-case Codabench results remain frozen
and may not be used for case-level tuning.

