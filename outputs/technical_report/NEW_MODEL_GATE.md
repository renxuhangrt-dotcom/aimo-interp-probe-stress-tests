# New-model research gate

Effective 2026-09-10. This is a hard project rule, not a suggestion.

## Frozen evidence that may not be tuned against

- V6's 19-case Codabench outcome, predictions, and inferred case labels.
- E10's ten problem IDs, texts, labels, individual predictions, and vote
  fractions.
- All public OOF predictions already inspected in E2 and E6–E9.

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
   source is available; this blocks V8.
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

## Eligible future hypothesis family

A potentially distinct direction is **perturbation-response geometry**: create
multiple label-blind, meaning-preserving variants of each problem and measure
how internal states or confidence trajectories change across variants. The
hypothesis is that robust reasoning produces stable relational geometry even
when surface form changes, whereas brittle solutions exhibit anisotropic or
localized state changes. This differs from predicting labels from one original
prompt state.

This is only a research hypothesis. It is not authorized for implementation
until an untouched, rule-compliant independent validation source is secured and
the exact transformation family is preregistered without reference to E10 cases.

