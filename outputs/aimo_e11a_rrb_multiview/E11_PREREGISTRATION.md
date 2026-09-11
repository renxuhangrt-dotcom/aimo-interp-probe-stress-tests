# E11a preregistration: controlled multi-view latent drift

Frozen on 2026-09-11 (Asia/Shanghai), after E11 stopped on an integrity check
and before any E11/E11a candidate metric or external robustness label was
observed.

## Repair boundary from E11

E11 successfully completed all 668 label-blind forward passes but stopped
before reporting candidate metrics because freshly retrained V6 scored
`0.676980198019802`, rather than reproducing its historical
`0.7047579757975797`. No external robustness labels were generated.

E11a changes only the control implementation:

- Stage-A paired comparisons use the immutable historical V6 OOF predictions.
- External V6 predictions use the exact deployed 225-vote artifact from public
  repository commit `c87824100d0570078efb21f564ea885bd4acd017`, ZIP SHA-256
  `C757708FCD5F079AF7A90EC30E91D45E5E9542FDBA32C125A862ACC68D933F3E`.
- Both controls are hash-checked before use.

The E11a hypothesis, features, labels, folds, seeds, classifier, thresholds,
and external validation protocol are unchanged. A new working directory and
cache namespace prevent reuse of E11 state.

## Motivation and independence

E10 showed that the V6 absolute hidden-state probe can acquire a severe
out-of-distribution positive bias. E11a tests a new representation: how the
same problem's hidden state moves under fixed, answer-preserving probe views.

The three feature views are fixed here and are not used to make the external
labels:

1. collapse all whitespace;
2. append a fixed, explicitly irrelevant sentence;
3. place the problem inside a neutral “copied verbatim” wrapper.

External labels use a disjoint source and disjoint transformations: the 30
AIME 2024 problems at Hugging Face revision
`2fe88a2f1091d5048c0f36abc874fb997b3dd99a`, transformed according to sentence
reversal, word reversal, and split reversal from Robust Reasoning Benchmark
(RRB) commit `0ae533b7bda703747089ffd1b5574fc010ea05f7`.

No E10 labels, hidden Codabench predictions, or leaderboard feedback may enter
E11a training, feature selection, threshold selection, or promotion.

## Fixed representation and classifier

- Target model: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`, revision
  `6e8885a6ff5c1dc5201574c8fd700323f23c25fa`, exact float16 weights.
- Prompt representation: final input token at layers
  `[4, 8, 12, 16, 20, 24, 28, 32, 36]`.
- For each probe view and layer: cosine distance and relative L2 distance from
  the original view.
- For each of the six view/metric combinations: mean over all layers, mean over
  the last three selected layers, and maximum.  Total: 18 fixed features.
- Classifier: training-fold-only standardization followed by balanced L2
  logistic regression, `C=0.1`, `liblinear`, fixed threshold `0.5`.
- Five problem-grouped folds for seeds `42, 43, 44, 45, 46`.
- The Stage-A V6 control is the frozen historical OOF prediction vector and
  must reproduce balanced accuracy `0.7047579757975797`. The external control
  is the hash-pinned deployed V6 artifact applied to the newly extracted
  original-view representations.

## Stage A: in-distribution promotion gates

All must pass before a single external answer is generated:

- grouped OOF balanced accuracy >= `0.7247579757975797` (V6 + 0.02);
- candidate problem-bootstrap 95% lower bound > `0.6169437631511935`;
- paired bootstrap 95% lower bound of `(E11a BA - V6 BA)` > `0`;
- each source-holdout balanced accuracy >= `0.65` and their mean >= `0.69`;
- advantage over the fixed shuffled-label control >= `0.10`;
- OOF disagreement versus V6 >= `0.08`.

Failure ends E11 as a negative result and no external labels are generated.

## Stage B: independent external validation

If Stage A passes, the notebook first writes and hashes the frozen E11 and V6
predictions for all 30 external problems.  Only then does it generate labels.

Generation is fixed to two samples per problem/view, temperature `0.6`, top-p
`0.95`, at most `1536` new tokens, and the RRB non-agentic reconstruction
protocol.  A problem is eligible when baseline accuracy is at least `0.5`.
Its robust label is true exactly when no one of the three perturbation
accuracies is below its baseline accuracy.

The external result is decision-grade only with at least 15 eligible problems
and at least four examples of each class.  All promotion gates must then pass:

- E11a ordinary accuracy >= best constant accuracy + `0.10`;
- E11a balanced accuracy >= `0.60`;
- E11a gets at least two more eligible problems correct than V6.

## Consequence

Passing both stages permits engineering a new Small-track submission artifact;
it does not automatically create or submit one.  Any failed gate records a
negative result and leaves V6 as champion.
