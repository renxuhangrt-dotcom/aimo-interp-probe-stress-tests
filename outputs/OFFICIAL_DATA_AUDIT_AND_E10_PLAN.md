# Official data audit and E10 decision

Date: 2026-09-10 (Asia/Shanghai)

## Outcome

The official augmentation files do not add independent supervised Small-model
training problems. The only defensible next Small step is E10: evaluate the
frozen V6 method on the direct `qwen3-8b:low` slice of the separate ten-problem
AIMO sample. No V8 is built before that result.

## Evidence

The pinned official baseline commit
`7e8839966750059b6b1d247a12ab56552c79b342` contains:

- `math-robust-agg.csv`: 141 rows, 137 unique problem IDs, with four exact
  label-consistent duplicate rows;
- `math-robust-final.csv`: 675 perturbation-family rows but still exactly 137
  unique problem IDs and 137 unique normalized original problem texts;
- the aggregate and full files have a 137/137 intersection by problem ID and
  by normalized original problem text;
- every full-file problem has 2–7 perturbation-family rows, so treating those
  rows as independent folds would leak the same problem across train and test;
- the binary rule `max(absolute_accuracy_decay) <= 0` reproduces all 137
  official aggregate labels (36 robust and 101 non-robust).

The current official Hugging Face organization lists the 141-row aggregate,
54-row filtered aggregate, 259-row filtered full data, 156-row augmentation,
28-row validation sample, and 558-row challenge sample. The augmentation pages
identify a single `qwen3-8b:low` model and the same Hendrycks-MATH sources used
by our 137 problems. These are filtered or expanded views, not an independent
labeled validation population.

The separate `aimo-interp-challenge-sample-full` snapshot has 558 rows, ten
unique AIMO problems, and eight model IDs. It includes direct
`qwen3-8b:low` rows and has zero intended overlap with the MATH training pool;
the notebook verifies the exact overlap rather than assuming it. The official
`problems-public` dataset has 22 symbolic-program problems but exposes no model
robustness labels, so it is not a supervised Small validation set.

## E10 execution

Run `aimo_e10_ood_small.ipynb` on Kaggle with T4 x2 and Internet enabled. The
notebook pins both the official dataset revision and Parquet hash, derives one
label per direct Small problem using the already validated rule, checks exact
independence, extracts the same final-prompt-token representation used by V6,
and produces `aimo_e10_ood_small_internals.zip`.

E10 is evaluation-only. Its individual problems will not become training
examples. The decision gate and minimum sample size were frozen in
`E10_PREREGISTRATION.md` before extraction.

## Strategic consequence

If E10 is decision-grade and passes, the next method experiment may be designed
from a new mechanistic hypothesis, but still cannot tune on E10. If E10 fails
or is too small, V6 remains the Small submission champion at 12/19 and Small
leaderboard iteration pauses. Work then shifts to the independent technical
report prize, where the audited duplicate leakage, failed-transfer experiments,
strict paired gates, and efficient zero-budget workflow are substantive
results rather than wasted effort.

## Official sources

- Challenge: https://aimo-interp.github.io/
- Official datasets: https://huggingface.co/aimo-interp
- Independent sample-full:
  https://huggingface.co/datasets/aimo-interp/aimo-interp-challenge-sample-full
- Public symbolic problems:
  https://huggingface.co/datasets/aimo-interp/problems-public
- Official baselines: https://github.com/aimo-interp/baselines

