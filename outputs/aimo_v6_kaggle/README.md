# Kaggle T4 x2 extraction edition

Use the corrected `aimo_v6_kaggle_v2.ipynb`; no Kaggle API token is required.

## Upload and queue

1. Open Kaggle and choose **Create → New Notebook**.
2. Use **File → Import Notebook**, then select `aimo_v6_kaggle_v2.ipynb`.
3. In **Settings**, set **Accelerator → GPU T4 x2**. Do not select P100.
4. Enable **Internet** so the pinned official repository and the 16.4 GB model
   can be downloaded.
5. Choose **Save Version → Save & Run All**. This queues a clean, reproducible
   run and avoids the 20-minute interactive-idle timeout.
6. When the run succeeds, open its **Output** panel and download
   `aimo_v6_internals.zip`.
7. Give that ZIP back to Codex. Local CPU validation will run before any V6
   artifact is created.

The notebook refuses single-GPU sessions, P100, CPU/disk model offload,
unexpected source data, incomplete extraction, non-finite states, and wrong
hidden-state dimensions. Per-problem caches live in `/kaggle/working`. If a run
is interrupted, attach that run's Output to the next notebook version as an
Input; the script automatically discovers and validates its `problem_cache`
files before resuming.

Expected output size is roughly 80–90 MB. The downloaded model cache stays in
Kaggle temporary storage and is deliberately excluded from notebook output.

## Important numerical note

T4 does not support BF16, so this edition uses the official baseline's FP16
fallback. It does not use 8-bit or 4-bit quantization. The manifest records the
dtype, model ID, official commit, data hash, dimensions, device map, and output
ZIP hash for the technical report.

Version 2 uses a platform-independent canonical-content hash. Version 1 compared
raw CSV bytes, so Linux and Windows newline differences caused a false data-drift
failure before extraction began.
