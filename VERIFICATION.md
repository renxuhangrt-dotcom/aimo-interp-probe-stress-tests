# Verification record

Verified on 2026-09-11 before the v0.1.0 local release candidate.

With the pinned official CSV files present, the V6, problem-token, and
E10 Kaggle extractor suites passed 16/16 tests, including all frozen
dataset/hash checks.

Inside the sanitized release without redistributed official data:

- core data/extraction/training pipeline: 5/5 passed;
- layer-vote synthetic signal: 1/1 passed;
- V6 Kaggle extractor: 5/5 runnable tests passed, 1 data-dependent hash test skipped;
- problem-token extractor: 5/5 runnable tests passed, 1 data-dependent hash test skipped;
- E10 extractor: 3/3 runnable tests passed, 1 data-dependent hash test skipped;
- report integrity: 3/3 passed.

Total release-local result: 22 passed, 3 explicitly skipped because the
official CSV files are intentionally not bundled. Supplying the pinned
official checkout activates those three strict hash tests. The release
builder separately compiles every included Python source and Notebook
code cell and scans text plus the V6 archive for credentials and local
user paths.
