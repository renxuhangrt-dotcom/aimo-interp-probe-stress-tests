# E12 preregistration: counterbalanced metacognitive robustness readout

Frozen on 2026-09-11 (Asia/Shanghai), after E11a failed Stage A and before any
E12 feature, candidate metric, or external robustness label was observed.

## New hypothesis

E11a established that generic hidden-state movement under harmless formatting
views is not a strong, source-stable robustness signal. E12 tests a different
hypothesis: when the target model is explicitly placed in a behavioral-
forecasting context, its layerwise preference between "stable" and "unstable"
descriptions contains robustness-specific information beyond a general estimate
of mathematical correctness.

This is a semantic readout experiment, not another distance-summary experiment.
No E11a feature is reused.

## Frozen diagnostic design

For every problem, the exact pinned model receives four prompts:

1. paraphrase robustness, with the stable description assigned to option A;
2. paraphrase robustness, with the stable description assigned to option B;
3. first-attempt correctness, with the correct description assigned to option A;
4. first-attempt correctness, with the correct description assigned to option B.

The system instruction says not to solve the mathematics. Each prompt ends with
the same assistant prefill, `The more likely description is option`. The next-
token logits for the single tokens ` A` and ` B` are read at layers
`[4, 8, 12, 16, 20, 24, 28, 32, 36]` using the model's frozen final
normalization and output head.

At each layer the A-minus-B margin is sign-corrected so a positive value always
means stable/correct. Reversing the option order therefore supplies an explicit
position-bias control. The two orders are averaged separately for robustness
and correctness. The fixed 18 features are:

- nine order-averaged robustness semantic margins;
- nine robustness-minus-correctness margins.

Order agreement, correlation, and absolute disagreement are reported for
diagnosis but are not promotion gates and cannot be used to alter the method.

## Fixed model and classifier

- Target model: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`, revision
  `6e8885a6ff5c1dc5201574c8fd700323f23c25fa`, exact float16 weights.
- Public labels: the audited 137-problem official aggregate at commit
  `7e8839966750059b6b1d247a12ab56552c79b342`.
- Classifier: training-fold-only standardization followed by balanced L2
  logistic regression, `C=0.1`, `liblinear`, fixed threshold `0.5`.
- Five problem-grouped folds for seeds `42, 43, 44, 45, 46`.
- Control: the immutable historical V6 OOF predictions with balanced accuracy
  `0.7047579757975797`.

No threshold, layer, feature, seed, classifier, or prompt may be changed after
observing E12 results. A changed hypothesis requires a new experiment number.

## Stage A: public grouped validation

Stage A performs exactly 548 diagnostic forward passes: four views for each of
137 public problems. It does not load the AIME 2024 external dataset, download
the deployed V6 artifact, extract external features, or generate external
answers.

All gates must pass:

- grouped OOF balanced accuracy >= `0.7247579757975797` (V6 + 0.02);
- candidate problem-bootstrap 95% lower bound > `0.6169437631511935`;
- paired bootstrap 95% lower bound of `(E12 BA - V6 BA)` > `0`;
- each source-holdout balanced accuracy >= `0.65` and their mean >= `0.69`;
- advantage over the fixed shuffled-label control >= `0.10`;
- OOF disagreement versus V6 >= `0.08`.

Failure is a valid negative result and ends the run.

## Stage B: untouched independent validation

Only after Stage A passes does the notebook load the 30 AIME 2024 problems,
extract their E12 features, and extract original-prompt states for the exact
deployed V6 control. Candidate and V6 external predictions are written and
hash-frozen before any robustness answer is generated.

Labels then use the disjoint RRB sentence-reversal, word-reversal, and
split-reversal protocol pinned at commit
`0ae533b7bda703747089ffd1b5574fc010ea05f7`. Generation remains two samples per
problem/view, temperature `0.6`, top-p `0.95`, and at most 1536 new tokens.

The result is decision-grade only with at least 15 eligible problems and four
examples of each class. All gates must pass:

- E12 ordinary accuracy >= best constant accuracy + `0.10`;
- E12 balanced accuracy >= `0.60`;
- E12 gets at least two more eligible problems correct than deployed V6.

Passing both stages permits a separate Small-track submission-engineering step;
it does not automatically create or upload a submission.
