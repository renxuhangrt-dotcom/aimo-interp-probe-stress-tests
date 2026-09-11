# Public cross-model transfer audit

This notebook extracts DeepSeek-R1-Qwen3-8B representations for the eight
unique problems in the official public multi-model sample. It does not train a
model, read public labels into the extraction output, build a submission, or
contact Codabench.

## Run on Kaggle

1. Import `aimo_public_transfer_audit_v2.ipynb` into a new Kaggle notebook.
2. Select **GPU T4 x2** and enable **Internet**.
3. Choose **Save Version → Save & Run All**.
4. After completion, download `aimo_public_transfer_internals.zip` from Output.
5. Return that ZIP to Codex for the frozen transfer audit.

Expected runtime is similar to the successful V6 extraction because model
download/loading dominates, while the actual seven-problem forward pass is
short. The notebook checks the public parquet SHA-256, expected 28 input rows,
eight unique problem texts, two-GPU placement, finite activations, dimensions,
and complete output before creating the ZIP.

Notebook SHA-256:
`CF87A5ADD7DB17A71ED1501931CE2D920F5DB10D6166B196B002CBA4955678E6`

The superseded v1 notebook expected seven unique problems and intentionally
failed before model loading when the frozen public file correctly contained
eight. Do not rerun v1.
