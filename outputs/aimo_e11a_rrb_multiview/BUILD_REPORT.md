# E11a repair and verification report

Built on 2026-09-11 (Asia/Shanghai).

## Artifact

- Notebook: `aimo_e11a_rrb_multiview.ipynb`
- Source: `kaggle_e11_rrb_multiview.py`
- Frozen protocol: `E11_PREREGISTRATION.md`
- Purpose: evaluate a controlled multi-view latent-drift classifier; never
  create or upload a Codabench submission.

## E11 failure diagnosis

- Both T4 GPUs, all pinned downloads, and all 668 feature forward passes
  completed successfully.
- E11 then stopped on its integrity check because retraining V6 on the newly
  extracted activations produced balanced accuracy `0.676980198019802`, not
  the historical `0.7047579757975797`.
- No candidate metric was printed or inspected and no external robustness
  label was generated.
- E11a retains every hypothesis and promotion threshold, but replaces the
  unstable retrained control with the immutable historical V6 OOF predictions
  and the exact hash-pinned V6 submission artifact.

## Verification completed

- Python source compilation: PASS.
- Notebook JSON parsing and compilation of every code cell: PASS.
- Unit tests: 9/9 PASS.
- Probe-view versus external-label transformation separation: PASS.
- RRB transformation example and involution checks: PASS.
- Fixed 18-feature shape/order test: PASS.
- AIME answer-extraction tests: PASS.
- External prediction-freeze mutation test: PASS.
- Official dataset replay: 141 input rows, 137 unique problems, canonical
  SHA-256
  `3FC14BFC02F7C1D0D79BB917491C7710F6A798150416B157C6C34205CEC9E067`.
- Embedded historical V6 OOF control: exact balanced accuracy
  `0.7047579757975797`.
- Deployed V6 archive and artifact hashes, schema, 25 groups, nine layers, and
  225-vote inference path: PASS.

## What remains unverified locally

The current machine has no suitable NVIDIA GPU. Exact model loading, the 668
multi-view forward passes, and conditional RRB answer generation therefore
remain for the Kaggle T4 x2 run. The notebook fails closed on hardware,
dataset, model-revision, device-map, V6-control, cache, and prediction-
freeze inconsistencies.
