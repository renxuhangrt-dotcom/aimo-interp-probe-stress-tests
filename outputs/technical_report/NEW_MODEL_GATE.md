# New-model research gate

Updated 2026-09-11. This is a hard project rule, not a suggestion.

## Frozen evidence that may not be tuned against

- V6's 19-case Codabench outcome, predictions, and inferred case labels.
- E10's ten problem IDs, texts, labels, individual predictions, and vote
  fractions.
- All public OOF predictions already inspected in E2 and E6–E9.
- E11a/E12 public metrics, prompts, features, option controls, and failure
  diagnostics.

These results may motivate a broad research question or reject a method family.
They may not choose a threshold, layer, feature, prompt rule, classifier,
ensemble weight, or per-case override.

## Conditions required before any V8-or-later implementation

All fields below must be completed in a dated preregistration before code is
trained or evaluated:

1. **Distinct mechanistic hypothesis.** State why the proposed signal should
   track response to meaning-preserving perturbations. "Another classifier,"
   "another layer," and "another pooling rule" over the existing final-token
   cache do not qualify.
2. **Independent validation source.** Name a labeled source not used to design
   the method, prove zero problem/text overlap, document model identity and
   perturbation semantics, and freeze its revision/hash. At present, no such
   The pre-designated AIME/RRB source remains untouched, but no candidate has
   passed the public Stage-A gate required to consume it.
3. **Rules compliance.** Do not use extra labeled data for the same final-test
   perturbation types. Unlabeled data or labeled data for genuinely different
   perturbations may be used only after documenting why the competition rules
   permit it.
4. **Primary metric alignment.** Ordinary accuracy is primary because it is the
   competition metric. Balanced accuracy, both constant baselines, class counts,
   coverage, and invalid predictions are mandatory diagnostics.
5. **Frozen comparison.** Predefine the baseline, minimum effect, paired test or
   interval, sample-size requirement, and action for pass/fail/inconclusive.
6. **No hidden-set feedback loop.** At most one confirmation submission may be
   assigned to a preregistered candidate. A private score cannot be converted
   into case labels or used for the next candidate.
7. **Reproducibility.** Pin data/model/code hashes, seeds, preprocessing scope,
   hardware, expected archive contents, and fail-closed integrity checks.

No version number, submission archive, or Codabench upload is authorized until
all seven conditions pass review.

## Current bottleneck and eligible future work

E11a has now tested perturbation-response geometry and failed Stage A; E12 has
tested counterbalanced metacognitive confidence and also failed Stage A. Their
features, prompts, distances, logit axes, layers, and controls are no longer
eligible tuning surfaces. The pre-designated AIME/RRB validation source remains
untouched because neither method reached Stage B, but its existence alone does
not justify another observational readout over the same 137 labels.

The current Small program is therefore paused at a technical bottleneck. A
future experiment is eligible only if it introduces a genuinely new causal
capability or information source—for example a pre-specified activation
intervention with a mechanistic prediction, or newly released independent
labeled problems. It must still satisfy all seven conditions above. No further
prompt, threshold, classifier, pooling, distance-summary, or option-token
variation is authorized.
