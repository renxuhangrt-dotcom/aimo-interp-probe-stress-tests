# E7 problem-token views notebook build report

Status: **READY FOR KAGGLE T4 x2 RUNTIME TEST**

- Notebook: `aimo_problem_token_views.ipynb`.
- Notebook SHA-256:
  `02DD2F904CEEBC512392234237AEFE885AB219FBCDF88F237484E1D303FC847D`.
- Extractor source SHA-256:
  `0B8983234F3BAD584258B96E11DD8B323ADDD9A66069FCF11185CCC284ED1646`.
- E7 preregistration SHA-256:
  `C07DC31F138914CA64821F29C96E7F8825953DE0FAC5CB0B1D4CD147E393DBE9`.
- Frozen E7 validation runner SHA-256:
  `F28ECC6493EEEFCD9988F6436A57EF2A52280462A366ACEB5BD08F70A9B59517`.
- Notebook format: v4.5, two cells; embedded code compiles.
- Extraction tests: 6/6 passed.
- Embedded-secret scan: clean.
- Official dataset reproduction: 137 unique problems; canonical SHA-256
  `3FC14BFC02F7C1D0D79BB917491C7710F6A798150416B157C6C34205CEC9E067`.
- CUDA runtime test: pending because the local machine has no NVIDIA GPU.

The extractor uses tokenizer offset mappings to locate the exact mathematical
problem span in the rendered chat prompt. It saves mean-pooled and final-token
representations for that span from all 37 hidden-state layers, while excluding
system and assistant-header tokens. Every cache entry is fingerprinted and
written atomically.

E7 evaluates only the preregistered `problem_mean` view. Promotion requires a
positive paired-bootstrap lower bound versus V6 in addition to all absolute,
source-transfer, control, effect-size, and disagreement gates.

