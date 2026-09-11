# E12 result analysis

## Decision

**FAIL_STAGE_A. Do not create V8 and do not tune E12.**

Returned result archive SHA-256:
`9FA60E9377F57EC62895965FB798BDC98DBAF14132286692A18544890C1F733C`.
The ZIP contains five valid entries and passes its CRC check.

## Integrity and containment

- Exact target revision:
  `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B@6e8885a6ff5c1dc5201574c8fd700323f23c25fa`.
- Exact public 137-row canonical hash reproduced:
  `3FC14BFC02F7C1D0D79BB917491C7710F6A798150416B157C6C34205CEC9E067`.
- Hardware was exactly 2 × Tesla T4 with no CPU or disk offload in the device map.
- All 548/548 preregistered diagnostic forwards completed; maximum input length
  was 344 tokens.
- The tokenizer confirmed distinct single tokens for ` A` and ` B` (IDs 362
  and 425).
- The run stopped before loading AIME 2024, downloading the deployed external
  V6 control, extracting external features, or generating RRB labels.

## Stage-A metrics

| Metric | E12 | Required |
|---|---:|---:|
| Grouped OOF balanced accuracy | 0.636276 | >= 0.724758 |
| Ordinary accuracy | 0.635036 (87/137) | diagnostic only |
| Bootstrap 95% lower bound | 0.540968 | > 0.616944 |
| Paired improvement lower vs V6 | -0.174262 | > 0 |
| Source holdout, test → train | 0.530000 | >= 0.65 |
| Source holdout, train → test | 0.556541 | >= 0.65 |
| Mean source holdout | 0.543271 | >= 0.69 |
| Shuffled-label control | 0.630363 | — |
| Advantage over shuffled control | 0.005913 | >= 0.10 |
| Disagreement versus V6 | 0.270073 | >= 0.08 |

E12 confusion counts were TN=64, FP=37, FN=13, TP=23, with 60 predicted
positives against 36 actual positives. Frozen V6 has TN=61, FP=40, FN=7,
TP=29 and 90/137 ordinary accuracy. E12 therefore gains three true negatives
but loses six true positives, for three fewer correct decisions overall.

## What failed

The counterbalancing control exposes a dominant option-position effect:

| Diagnostic | Semantic sign agreement after A/B reversal | Pearson correlation | Mean absolute margin difference |
|---|---:|---:|---:|
| Robustness | 0.287105 | 0.117567 | 2.759150 |
| Correctness | 0.083536 | -0.770602 | 3.786645 |

If the readout represented the option meanings consistently, sign-corrected
margins from the two option orders would agree. Instead, correctness margins
almost reverse when A and B are swapped. Averaging the orders removes much of
the position preference but leaves little reproducible label signal. The
near-zero advantage over a shuffled-label control and near-chance cross-source
scores independently confirm that this is not a useful robustness estimator.

E12 is also not materially better than E11a: ordinary accuracy remains exactly
87/137, and the confusion matrix merely trades one false positive for one true
positive relative to E11a. The mechanism changed; the transferable result did
not.

## Consequence for the score-first workflow

Do not change the prompt wording, assistant prefill, A/B tokens, layer set,
classifier, regularization, or threshold after observing this result. Such a
run would be post-hoc tuning on the same 137 labels.

The agreed technical-bottleneck condition is now met for the current Small
optimization program:

- behavioral answer-stability variants plateaued at 0.53 privately;
- V7's locally improved calibration fell from V6's 12/19 to 9/19 privately;
- E6–E9 produced no preregistered, statistically credible improvement over V6;
- V6 collapsed on the independent E10 OOD audit;
- two genuinely new mechanisms, E11a multi-view drift and E12 metacognitive
  readout, both failed Stage A before independent labels were touched.

V6 remains the frozen Small champion at 12/19. Further Small submissions are
not justified without a genuinely new information source or experimental
capability—for example new independent labeled problems, causal interventions
rather than observational readouts, or awarded compute support. The rational
next action under the user's stopping rule is to consolidate the technical
report and preserve the still-untouched AIME/RRB source for a future method
that first demonstrates credible public Stage-A gains.
