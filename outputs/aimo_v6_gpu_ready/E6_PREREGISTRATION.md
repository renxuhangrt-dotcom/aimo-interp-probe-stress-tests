# E6 preregistration: adjacent-layer representation-change vote

Frozen before E6 execution on 2026-09-10 (Asia/Shanghai).

## Motivation

V7 showed that stricter calibration of the same raw-layer probes does not
transfer. E6 therefore changes the representation rather than the decision
threshold. For each four-layer stage it uses the final-prompt-token change
`h_l - h_(l-4)`, which may suppress static prompt/source information and retain
how the model transforms the problem representation through depth.

## Fixed method

- Input: the existing label-blind float32 activation cache; no GPU extraction.
- Deltas: `4-0`, `8-4`, `12-8`, `16-12`, `20-16`, `24-20`, `28-24`, `32-28`,
  and `36-32`.
- One balanced L2 logistic probe per delta, fixed `C=0.001`.
- Each probe is standardized using its training partition only.
- Nine equal hard binary votes; fixed final threshold `0.5`.
- Five problem-grouped folds for seeds `42..46`.
- Both source-holdout directions and the seed-42 shuffled-label control repeat
  the identical delta construction and classifier.
- Uncertainty: 5,000 problem-level bootstrap samples, seed `20260910`.
- V6/E2 OOF predictions are used only for a paired aggregate comparison. No
  hidden Codabench IDs, predictions, inferred labels, or V7 thresholds enter
  training or selection.

## Frozen promotion gates

All must pass:

- grouped OOF balanced accuracy >= V6 + `0.02` (>= `0.7247579757975797`);
- candidate bootstrap 95% lower bound > V6's `0.6169437631511935`;
- paired bootstrap 95% lower bound of `(E6 BA - V6 BA)` > `0`;
- each source-holdout BA >= `0.65` and their mean >= `0.69`;
- advantage over shuffled-label control >= `0.10`;
- OOF disagreement versus V6 >= `0.08`.

Failure of any gate records E6 as negative and produces no submission archive.
Passing all gates permits artifact engineering, but not automatic Codabench
submission.

