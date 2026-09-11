# AIMO V6 GPU-ready validation package

This package prepares the next trained-probe experiment without creating a V6
submission.  It is fail-closed: the final command exits with code `2` when any
scientific gate fails.

## What changed from the official baseline

The official baseline at commit `7e8839966750059b6b1d247a12ab56552c79b342`
uses row-wise `StratifiedKFold`.  The public enhanced dataset contains repeated
rows for a few `problem_id` values, so row-wise splitting can place the same
problem in training and validation.  This package first collapses exact,
consistent duplicates and then enforces `problem_id`-grouped splits everywhere.

The validation protocol is fixed before seeing new hidden-state results:

- 137 unique public problems, with labels and the two official source IDs kept;
- one forward-only extraction per unique prompt, using the official system prompt;
- 5 outer folds × 5 seeds, with layer and `C` selected only inside each outer fold;
- both train-source → test-source holdout directions;
- one shuffled-label nested-CV control;
- 5,000 grouped bootstrap resamples;
- no artifact export unless every threshold passes.

## Run on the allocated Linux GPU node

From the project root:

```bash
bash outputs/aimo_v6_gpu_ready/setup_gpu_env.sh
export PYTHON_BIN="$PWD/outputs/aimo_v6_gpu_ready/.venv-gpu/bin/python"
export HF_TOKEN="your-read-token"  # only in the shell; never save it in this repo
GPU_COUNT=2 bash outputs/aimo_v6_gpu_ready/run_gpu.sh
```

`run_gpu.sh` pins and checks the official baseline commit, regenerates the
deduplicated dataset, runs one resumable extraction shard per GPU, consolidates
all layers, and runs the preregistered validation.  Re-running the command reuses
valid per-problem cache files.  Extraction logs are under `work/v6_gpu_run/`.

The decision is written to `latest_validation/VALIDATION_REPORT.md` and the full
machine-readable evidence to `latest_validation/validation_result.json`.

## PASS conditions

- nested OOF balanced accuracy ≥ 0.65;
- grouped bootstrap 95% lower bound > 0.55;
- each source-holdout balanced accuracy ≥ 0.60;
- real-label score exceeds the shuffled-label control by ≥ 0.05.

If the result is FAIL, keep the cached internals for diagnosis, but do not name,
package, or upload anything as V6.
