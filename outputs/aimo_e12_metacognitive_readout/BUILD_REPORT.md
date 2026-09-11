# E12 build report

Built: 2026-09-11 (Asia/Shanghai)

## Outcome

The Kaggle notebook and inspectable Python source implement the frozen E12
counterbalanced metacognitive-readout experiment. This package does not contain
a Codabench submission and does not claim an experimental result.

## Workflow safeguards

- Stage A has only three inputs: official rows, official diagnostic margins,
  and the frozen feature-name list.
- Stage A executes 548 public-data forward passes (137 problems × four
  counterbalanced diagnostic prompts).
- The AIME 2024 external dataset, deployed V6 external artifact, external
  feature passes, and RRB generations are reached only after Stage A passes.
- External E12 and V6 predictions are hash-frozen before any external
  robustness generation.
- A/B option order is reversed and the semantic margin sign is corrected before
  the two orders are averaged.
- A multi-GPU-safe readout moves each layer vector to the final-normalization
  device before stacking.

## Verification

- 12 local unit/integrity tests passed.
- Both Python files compile.
- The notebook is valid nbformat 4, GPU enabled, Internet enabled, and all code
  cells compile. Its executable code exactly matches the inspectable Python
  source byte-for-byte after notebook decoding.
- The embedded frozen V6 OOF vector reproduces balanced accuracy
  `0.7047579757975797`.
- The deployed V6 archive test reproduces 25 groups and 225 votes from the
  locally pinned archive.

Local tests do not execute the 8B model; the exact model-device and tokenizer
checks deliberately run fail-closed on Kaggle.

## Frozen file hashes (SHA-256)

- `kaggle_e12_metacognitive_readout.py`:
  `BEFF00CAC63E8580866F6E3619790711DE87C9E82D7AC3924AC2A0F4C92F57C6`
- `aimo_e12_metacognitive_readout.ipynb`:
  `D1D22C1CA7D0FEC317D00732E72DB98A4ED47557D92351D36F88F816A0E4CB95`
- `E12_PREREGISTRATION.md`:
  `3470AAC3564D42CC262233EA7E451B94622F7FA8A2601AEEE8BA2D017EA517BF`
- `test_e12_metacognitive_readout.py`:
  `8EEDDC9CA3F8725C2F3C0D6B57653CF0BDEC33586724128BC4EE7C5CA74A1571`
