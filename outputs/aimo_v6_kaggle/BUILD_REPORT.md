# Kaggle edition build report

Status: **READY FOR KAGGLE T4 x2 RUNTIME TEST**

- Notebook: `aimo_v6_kaggle_v2.ipynb`
- Notebook version: 2 (platform-independent dataset hash)
- Notebook SHA-256: `D2024085EE929C3D7F30FC62E0B96F960268578F0F0F70A0369086088E517400`
- Notebook format: v4.5, two cells (instructions + self-contained extractor)
- Embedded secret scan: clean
- Python compilation: passed
- Unit tests: 6/6 passed
- Frozen source-data reproduction: 137 problems and canonical-content SHA-256 passed
- CUDA runtime test: pending, because the local machine has no NVIDIA GPU

The runtime asserts two GPUs, rejects P100 and CPU/disk offload, verifies all
37 × 4096 hidden-state matrices, saves each problem atomically, imports valid
caches from an attached previous Kaggle output, and packages only completed
internals. It never produces a trained artifact or submission ZIP.

Version 1 stopped after 22.7 seconds because it compared OS-dependent raw CSV
bytes. Version 2 normalizes embedded line endings and validates the logical row
content instead. The failed run did not begin model loading or extraction.
