# When Good Probes Fail: Pre-registered Stress Tests of an Efficient Hidden-State Robustness Predictor for Mathematical Reasoning

**AIMO Interpretability Challenge 2026 — Small Models Track**  
**Technical-report candidate, version 0.3 (2026-09-11)**
**Author:** XUHANG REN · **Affiliation:** Independent Researcher  
**Contact:** renxuhang2020@qq.com  
**Code:** https://github.com/renxuhangrt-dotcom/aimo-interp-probe-stress-tests

## Abstract

Can a low-cost probe of a reasoning model's hidden states predict whether a
correct answer will survive meaning-preserving perturbations? We study this
question for `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` using 137 unique labeled
Hendrycks-MATH problems from the AIMO Interpretability Challenge. Our strongest
submission, V6, applies balanced linear probes to the final prompt-token state
at nine fixed layers and aggregates 225 hard votes. V6 reached 0.705 grouped
out-of-fold balanced accuracy and 12/19 accuracy on the Small Track private
evaluation, improving over our constant and earlier probe submissions while
requiring only a single model forward pass per problem and a 4.23 MB submission
bundle. Four pre-registered representation/classifier variants failed strict
paired promotion gates despite several small in-distribution improvements.
Two mechanism-changing follow-ups—multi-view latent drift and counterbalanced
metacognitive readout—then performed near their shuffled-label controls and
failed before independent labels were touched. Most importantly, a
frozen independent audit on ten non-overlapping AIMO problems found that V6
predicted every case robust, scoring 0.20 accuracy against an 0.80 always-negative
baseline. The result shows that probe separability and even a small private-set
gain did not establish out-of-distribution reliability. We contribute an
auditable low-resource workflow, a sequence of informative negative results,
and practical recommendations for evaluating interpretability methods under
small, imbalanced, sequentially observed datasets.

## 1. Motivation

The AIMO Interpretability Challenge asks whether a system can distinguish
robust mathematical reasoning from solutions that fail under controlled
counterfactual variation [1]. This is explicitly a generalization problem:
performance on an original problem does not reveal whether a model used a
stable mechanism or a brittle shortcut. The official baseline therefore
includes probes over internal representations, including final input-token
hidden states [2,3].

Probing is attractive to compute-limited participants because a frozen model
can be encoded once and the downstream classifiers are inexpensive. However,
probe accuracy is correlational and can reflect dataset regularities or probe
capacity rather than a mechanism used by the model [4,5]. Our central research
question is consequently not only whether a probe performs above a baseline,
but whether that advantage survives changes in problem source and evaluation
distribution.

## 2. Data audit and leakage control

The official aggregate file contains 141 rows but only 137 unique problem IDs.
Four rows are exact, label-consistent duplicates. We collapsed these before any
split and used the 137 unique problems: 101 non-robust and 36 robust. The two
sources contain 85 train-reference and 52 test-reference MATH problems.

The official 675-row expanded file does not add independent problems: it
contains the same 137 IDs and normalized original texts, with 2–7 perturbation
families per problem. Treating these rows as independent folds would leak the
same problem across train and validation. We verified that the published binary
label is reproduced for all 137 problems by

\[
y_i = \mathbb{1}\!\left[\max_{p \in P_i}
\mathrm{absolute\_accuracy\_decay}_{i,p} \leq 0\right].
\]

Every validation split was grouped by `problem_id`. Preprocessing, including
standardization and PCA where applicable, was fit inside the training partition
only. Exact dataset revisions, canonical hashes, and artifact hashes were
recorded before evaluation.

## 3. Frozen V6 method

For each original problem, we render the official solve prompt and extract the
final input-token hidden state from the frozen 8B checkpoint. V6 uses layers
4, 8, 12, 16, 20, 24, 28, 32, and 36. At each layer, a balanced L2 logistic
probe with fixed `C=0.001` is trained after training-partition-only
standardization.

We use five problem-grouped folds under five fixed seeds (42–46). Deployment
retains the 25 fitted fold groups at each of nine layers, producing 225 equal
binary votes. The final prediction is a fixed majority vote at 0.5. Randomized
label controls are evaluated but excluded from deployment. This design uses no
generation, external API, or Internet access during scoring. Its 4,225,167-byte
submission bundle contains only the competition interface, inference code,
track marker, and a compressed copy of the 8.32 MB probe artifact.

## 4. Validation protocol

V6 used repeated grouped out-of-fold validation, both directional
source-holdouts, a shuffled-label control, and 5,000 problem-level bootstrap
samples. Because earlier exploratory results had already been observed, V6's
public validation is described as **sequential exploratory evidence**, not an
untouched confirmation.

Experiments E6–E9 each changed one component and were registered before their
first evaluation. Promotion required all of the following: at least +0.02
balanced accuracy over V6; a candidate bootstrap lower bound above V6's; a
strictly positive paired-bootstrap lower bound; at least 0.65 on each source
direction and 0.69 on their mean; at least +0.10 over the shuffled-label
control; and at least 0.08 prediction disagreement from V6. These gates were
intended to prevent leaderboard-driven or negligible changes from becoming new
submissions.

After the independent E10 audit, E11a and E12 tested two genuinely different
mechanisms under the same fail-closed Stage-A gates. E11a measured hidden-state
drift under three answer-preserving prompt views. E12 directly read layerwise
preferences between robustness/correctness descriptions while reversing A/B
assignments to expose option-position bias. Both notebooks were programmed to
stop before loading the untouched AIME/RRB source unless every public gate
passed.

## 5. Results: useful signal, no validated successor

![Summary of public and OOD results](results_overview.svg)

| Experiment | Representation / classifier | OOF balanced accuracy | OOF accuracy | ΔBA vs V6 | Source holdouts | Random-control advantage | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| V6 | Final prompt token, linear votes | 0.7048 | 0.6569 | — | 0.6750 / 0.7073 | +0.1189 | Passed original exploratory gate |
| E6 | Adjacent-layer deltas, linear votes | 0.7137 | 0.6569 | +0.0089 | 0.6750 / 0.6619 | +0.1507 | Reject |
| E7 | Mean problem-token state, linear votes | **0.7186** | 0.6642 | +0.0139 | 0.6117 / 0.7561 | +0.1983 | Reject |
| E8 | Final problem-token state, linear votes | 0.6949 | 0.6423 | −0.0099 | 0.6583 / 0.6619 | +0.1170 | Reject |
| E9 | Final prompt token, PCA10 + RBF votes | 0.7126 | **0.7080** | +0.0078 | 0.6283 / 0.7683 | +0.1210 | Reject |
| E11a | Multi-view latent drift | 0.6273 | 0.6350 | −0.0774 | 0.5900 / 0.7173 | +0.0337 | Reject at Stage A |
| E12 | Counterbalanced metacognitive readout | 0.6363 | 0.6350 | −0.0685 | 0.5300 / 0.5565 | +0.0059 | Reject at Stage A |

V6's public confusion counts were TN=61, FP=40, FN=7, and TP=29. Its bootstrap
95% lower bound was 0.6169, and it exceeded the shuffled-label control by
0.1189 balanced-accuracy points. On the 19-case Small Track private evaluation,
V6 scored 12/19 (0.6316), with full coverage and no invalid predictions. The
derived private confusion counts were TN=5, FP=4, FN=3, and TP=7. This is
evidence that the method was not merely a constant predictor on that sample,
but the sample is too small to establish reliable generalization.

The rejected experiments are scientifically informative:

- **E6—layer changes:** A +0.0089 apparent gain changed only 2/137 decisions,
  correcting one V6 error and breaking one. Its paired interval crossed zero.
- **E7—semantic token pooling:** The highest public balanced accuracy had the
  strongest randomized-control advantage, yet its source directions diverged
  sharply (0.6117 vs 0.7561). A representation can expose label information
  without transferring symmetrically.
- **E8—final problem token:** Excluding the assistant header did not improve
  performance; both aggregate and transfer evidence weakened.
- **E9—nonlinear boundary:** Ordinary OOF accuracy increased to 0.7080, but the
  paired balanced-accuracy interval crossed zero and one source direction fell
  to 0.6283. This exposes a real metric tension under the 101:36 imbalance:
  competition accuracy and class-balanced diagnostic quality need to be
  reported together rather than silently substituted for one another.
- **E11a—multi-view drift:** The candidate disagreed with V6 on 25/137 rows,
  but its 0.6273 balanced accuracy was only 0.0337 above a frozen shuffled-label
  control. Source holdouts diverged to 0.5900 and 0.7173.
- **E12—metacognitive readout:** Reversing A/B assignments exposed a dominant
  position effect. Sign-corrected correctness margins agreed on only 8.35% of
  row-layer pairs and correlated at −0.771 across option orders. After
  counterbalancing, candidate performance was only 0.0059 above its shuffled
  control and both source holdouts were near chance.

## 6. Independent OOD audit

E10 evaluated the already frozen V6 artifact on the direct `qwen3-8b:low`
slice of a separately published ten-problem AIMO sample [6]. The pinned source
contained 558 rows across ten problems and eight models. The Small-model slice
contained 64 perturbation rows over all ten problems. We applied the label rule
validated on 137/137 training problems, found eight non-robust and two robust
problems, and verified zero overlap with training by both ID and normalized
text. Labels were not supplied to either the model forward pass or V6.

Before extraction, we registered a minimum of eight problems, representation
and artifact hashes, metrics, and a gate requiring accuracy ≥0.60 and at least
+0.10 over the best constant predictor. E10 was therefore decision-grade.

V6 predicted all ten problems robust. Accuracy was 0.20, balanced accuracy was
0.50, and the 95% Wilson interval for accuracy was [0.0567, 0.5098]. The
always-negative baseline scored 0.80. This was not a threshold near-miss: the
minimum positive-vote fraction was 0.7689 and the median was 0.94. We interpret
this as evidence of severe positive bias under the new problem source, or a
shortcut shared by this representation/classifier family. With only ten cases,
E10 does not estimate the population effect precisely; it does decisively fail
the registered continuation gate.

We did not inspect individual errors to construct rules, move thresholds, or
select layers, and we did not produce V8.

## 7. Actionable lessons

1. **Deduplicate and split at the causal unit.** Expanded perturbation rows are
   repeated measurements of a problem, not independent examples.
2. **Controls are necessary but insufficient.** Every candidate beat its
   randomized-label control, yet none established a stable improvement and V6
   still failed OOD.
3. **Small in-distribution gains require paired evidence.** E6–E9 changed
   decisions only modestly or had paired intervals crossing zero.
4. **Source-direction symmetry is a useful warning signal.** E7 and E9 looked
   competitive in aggregate while failing one transfer direction.
5. **Measure confidence under shift.** V6's uniformly high OOD vote fractions
   revealed confidently wrong extrapolation, not uncertainty that could be
   repaired by a small threshold adjustment.
6. **Pre-registration is an engineering control against overfitting.** Frozen
   gates converted tempting marginal gains into documented negative results and
   prevented repeated hidden-set tuning.
7. **Counterbalance elicited judgments.** A direct self-forecast can appear
   semantic while actually tracking answer position; reversing option order
   made this failure visible in E12.

## 8. Limitations and claim boundary

The training set contains only 137 unique problems from two related MATH
sources, and positive labels are a minority. The private result contains 19
cases; E10 contains ten. Neither supports a precise estimate across all AIMO
models, perturbations, or mathematical domains. E10 is a new problem source but
uses the same 8B checkpoint, so it tests problem transfer rather than model-scale
transfer. Linear separability does not establish that the model uses the probed
features causally. Finally, sequential exploration before V6 weakens the
confirmatory status of its public score. E11a and E12 failed before reaching
their independent Stage B, so they provide strong negative public evidence but
no additional OOD estimate. Our strongest defensible claim is
therefore negative: this efficient final-token probe contains in-distribution
signal, but the available evidence does not support it as a general detector of
robust mathematical reasoning.

## 9. Reproducibility and resource use

All datasets, code revisions, experiment registrations, activation schemas,
model artifacts, and result JSONs are hash-pinned. Every candidate used fixed
seeds and problem-grouped splits. The exact deployment scorer replayed all 137
OOF predictions before packaging. E10 failed closed on revision drift, overlap,
label-rule mismatch, incomplete extraction, non-finite states, or CPU/disk model
offload. The artifact manifest and machine-readable result tables accompany the
report.

The public implementation, preregistrations, frozen result files, and exact V6
submission bundle are available at
https://github.com/renxuhangrt-dotcom/aimo-interp-probe-stress-tests.

The work incurred zero direct monetary compute cost: activation extraction used
free Kaggle T4×2 sessions, while probe training, bootstrap evaluation, packaging,
and audit ran locally on CPU. This demonstrates that rigorous interpretability
experiments are possible for an independent participant without local NVIDIA
hardware, although the provided GPU access remains real compute rather than
"zero compute."

## 10. Conclusion

An efficient multi-layer probe achieved meaningful public and small-private-set
performance, but plausible refinements did not survive strict paired promotion
rules, the frozen method collapsed on an independent AIMO problem source, and
two mechanism-changing successors performed near randomized controls. The gap
between separability, elicited confidence, and transfer is the main result.
Under our pre-registered stopping logic, this accumulated evidence constitutes
a technical bottleneck rather than an invitation to tune further on 137 labels.
Future work requires a genuinely new causal capability or information source;
the V6 private set and E10 labels remain permanently excluded from tuning.

## References

1. Štefánik et al. **AIMO Interpretability Challenge.** NeurIPS 2026 Competition proposal, arXiv:2607.13899. https://arxiv.org/abs/2607.13899
2. AIMO Interpretability Challenge website. https://aimo-interp.github.io/
3. AIMO Interpretability Challenge baseline implementation. https://github.com/aimo-interp/baselines
4. Belinkov. **Probing Classifiers: Promises, Shortcomings, and Advances.** *Computational Linguistics* 48(1), 2022. https://aclanthology.org/2022.cl-1.7/
5. Hewitt and Liang. **Designing and Interpreting Probes with Control Tasks.** EMNLP-IJCNLP 2019. https://aclanthology.org/D19-1275/
6. AIMO Interpretability Challenge sample-full dataset. https://huggingface.co/datasets/aimo-interp/aimo-interp-challenge-sample-full
