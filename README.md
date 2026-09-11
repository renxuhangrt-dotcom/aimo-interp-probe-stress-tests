# When Good Probes Fail

Companion implementation and evidence for **When Good Probes Fail:
Pre-registered Stress Tests of an Efficient Hidden-State Robustness Predictor
for Mathematical Reasoning**, by XUHANG REN (Independent Researcher).

This local release candidate documents the AIMO Interpretability Challenge
2026 Small Models Track V6 method and the E6–E10 negative-result sequence.
The scientific result is not that V6 is a generally reliable detector: V6
reached 12/19 on the Small private evaluation but failed a pre-registered,
ten-problem zero-overlap OOD audit at 0.20 accuracy versus an 0.80 constant
baseline.

## Layout

- `outputs/technical_report/`: typeset two-page PDF, compact and full report
  drafts, figure, tables, governance rules, and reproducibility manifest.
- `outputs/aimo_v6_gpu_ready/`: data audit, hidden-state extraction, V6 training
  and export, E6–E10 experiments, preregistrations, tests, and aggregate results.
- `outputs/aimo_v6_kaggle/`: label-blind Kaggle extraction notebooks and sources.
- `outputs/aimo-small-v6-fixed-layer-vote-20260910.zip`: exact 4.23 MB submitted V6 bundle.
- `third_party/`: license and provenance notices for the official baseline.

Large activation arrays, model weights, official dataset copies, private
downloads, and individual E10 internals are intentionally not redistributed.

## Verify the report evidence

The report tables and vector figure are regenerated directly from frozen JSON;
the PDF is then typeset from the audited report assets:

```bash
python outputs/technical_report/build_report_assets.py
python outputs/technical_report/test_report_integrity.py
python outputs/technical_report/build_two_page_pdf.py
```

## Reproduce the V6 pipeline

1. Obtain the official baseline repository at commit
   `7e8839966750059b6b1d247a12ab56552c79b342`.
2. Run `outputs/aimo_v6_kaggle/aimo_v6_kaggle_v2.ipynb` on Kaggle T4 x2 to
   create the audited 137-problem hidden-state archive.
3. Extract the archive locally and run:

```bash
python outputs/aimo_v6_gpu_ready/train_layer_vote_probe.py   --internals PATH_TO_INTERNALS   --output-dir work/validation_e2_layer_vote   --layers 4 8 12 16 20 24 28 32 36
```

Artifact export requires the official solve prompt and a passing validation
JSON; see `export_layer_vote_artifact.py`. Exact deployment replay requires
regenerating the omitted activation internals.

## Integrity

- V6 archive SHA-256: `C757708FCD5F079AF7A90EC30E91D45E5E9542FDBA32C125A862ACC68D933F3E`
- All public-release files are enumerated in `MANIFEST.json`.
- `BUILD_AUDIT.json` records secret/path scanning, compilation checks, and the
  deterministic release ZIP hash.

## License

XUHANG REN's original additions are released under Apache-2.0; see `LICENSE`
and `NOTICE`. The official baseline is also Apache-2.0, and its license is
preserved separately under `third_party/aimo-baselines/`. Model and dataset
licenses remain governed by their respective upstream providers.
