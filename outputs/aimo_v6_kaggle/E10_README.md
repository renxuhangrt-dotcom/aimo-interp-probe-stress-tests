# E10 independent Small-model OOD audit

Use `aimo_e10_ood_small.ipynb`. This is a frozen evaluation extraction, not a
submission builder.

1. Import the notebook into Kaggle.
2. Select **GPU T4 x2** and enable **Internet**.
3. Choose **Save Version -> Save & Run All**.
4. After success, open **Output** and download
   `aimo_e10_ood_small_internals.zip`.
5. Return that ZIP to Codex for the preregistered V6 scoring audit.

Expected compute time is about 4–7 minutes after the GPU starts. There are at
most ten forward passes, so model download and loading dominate. The notebook
fails closed on dataset revision/hash drift, an incorrect GPU topology,
training/OOD problem overlap, a label-rule mismatch, CPU/disk model offload,
non-finite activations, or incomplete extraction.

The output contains public labels for aggregate scoring, but the forward pass
is label-blind. E10 problems must not be used for later per-case or threshold
tuning.

## Frozen-file checks

- Notebook SHA-256:
  `70A0FDC4F280400D588A7B96990B1A4C9BC82B84D3520A1914B65CAD0CB645F1`
- Extractor source SHA-256:
  `6A14966292BE3E486BB57B33ED88E9156DCDD83991B429AFCE58A9026606D269`
- Preregistration SHA-256:
  `D4DA08BACB7379D3AA5669061B4D9FCEA8881D23F5AB927C492665726F9C8BED`

The notebook/source synchronization and code compilation checks passed. The
extractor test suite passed 4/4 tests, and the frozen V6 scorer passed 3/3 tests,
including an exact 225-vote replay on the earlier eight-problem public audit.
