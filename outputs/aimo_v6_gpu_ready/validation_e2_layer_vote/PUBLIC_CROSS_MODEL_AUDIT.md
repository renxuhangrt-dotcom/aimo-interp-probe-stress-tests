# Frozen V6 public cross-model transfer audit

Decision: **NO-GO for a direct Main Track proxy submission**.

This is a diagnostic-only audit. No V6 weights, layers, votes, or thresholds were changed.

| Slice | Cases | Accuracy | Balanced accuracy | Always-false accuracy |
|---|---:|---:|---:|---:|
| All public multi-model cases | 24 | 0.3750 | 0.5000 | 0.6250 |
| Excluding qwen3-8b training-model case | 23 | 0.3913 | 0.5000 | 0.6087 |

All 24 cases were predicted positive. The positive-vote fractions span `0.7689` to `1.0000` with median `0.9778`.

## Per model

| Model ID | Cases | Positive labels | Positive predictions | Accuracy |
|---|---:|---:|---:|---:|
| `Qwen/Qwen3.5-397B-A17B-FP8:low` | 5 | 1 | 5 | 0.2000 |
| `gemini-3.1-pro-preview:low` | 3 | 0 | 3 | 0.0000 |
| `gpt-5.2-2025-12-11:low` | 3 | 3 | 3 | 1.0000 |
| `gpt-oss-120b:low` | 5 | 0 | 5 | 0.0000 |
| `huikang-gpt-oss-120b-aimo3:low` | 3 | 1 | 3 | 0.3333 |
| `lukealonso/GLM-5.1-NVFP4:low` | 4 | 4 | 4 | 1.0000 |
| `qwen3-8b:low` | 1 | 0 | 1 | 0.0000 |

## Limitations

- The public problem texts are evaluated with frozen DeepSeek representations; labels for the other model IDs were not used for training.
- The public sample is tiny and its model IDs do not exactly match every currently cached evaluation checkpoint.
- This audit freezes the V6 artifact and threshold; it must not be used for per-case or leaderboard-driven tuning.
