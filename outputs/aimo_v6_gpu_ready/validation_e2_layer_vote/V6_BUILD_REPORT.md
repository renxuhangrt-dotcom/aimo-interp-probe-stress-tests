# V6 build report

Date: 2026-09-10 (Asia/Shanghai)

## Decision

`PASS` — the preregistered E2 validation gates, artifact gate, deployment replay,
and submission-package checks all passed. The V6 archive was built locally and
was not uploaded.

E2 is sequential exploratory evidence: E1 source-holdout results were visible
before E2 was designed. The hidden Codabench run should therefore be treated as
the next independent check, not as a guaranteed score.

## Frozen method

- Model: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`
- Representation: final prompt-token hidden state
- Layers: `4, 8, 12, 16, 20, 24, 28, 32, 36`
- Linear probe: balanced L2 logistic regression, `C=0.001`
- Validation: 5 grouped folds × seeds `42, 43, 44, 45, 46`
- Deployment: 25 eligible fold groups × 9 layers = 225 equal binary votes
- Randomization groups: excluded from deployment

## Validation evidence

| Check | Result | Gate |
|---|---:|---:|
| Grouped OOF balanced accuracy | 0.704758 | >= 0.65 |
| Bootstrap 95% lower bound | 0.616944 | > 0.55 |
| Test-reference → train-reference | 0.675000 | >= 0.60 |
| Train-reference → test-reference | 0.707317 | >= 0.60 |
| Shuffled-label balanced accuracy | 0.585809 | diagnostic |
| Real-control advantage | 0.118949 | >= 0.05 |

Deployment-code replay covered all 137 public problems and reproduced the E2
OOF balanced accuracy exactly: `0.7047579757975797`.

## Artifact and archive

- Artifact schema: 3 (`fixed_multilayer_vote_probe_ensemble`)
- Artifact bytes: 8,316,999
- Artifact SHA-256: `7A4DF77D031530DC4998CBC4B6FFF2B8B717F99C7FA04E2646847C6A5BE3672B`
- Archive: `aimo-small-v6-fixed-layer-vote-20260910.zip`
- Archive bytes: 4,225,167
- Archive SHA-256: `C757708FCD5F079AF7A90EC30E91D45E5E9542FDBA32C125A862ACC68D933F3E`

The ZIP has exactly these root-relative entries:

```text
solution.py
probe_inference.py
small.txt
probe_artifacts/probe_artifact.pkl
```

`small.txt` is zero bytes. No cache, manifest, model weights, secrets, or extra
directory nesting is included.

## Verification

- 5/5 ingestion contract tests passed
- 3/3 track tooling tests passed
- 7/7 trained-probe safety tests passed
- 5/5 artifact-gate tests passed
- 1/1 fixed-layer-vote synthetic-signal test passed
- Artifact gate passed with no warnings or failures
- Extracted-ZIP deployment replay passed

The full unrelated repository test discovery contains an uncertainty-profiling
suite that imports PyTorch; PyTorch is not installed in this local audit venv.
The submission-specific tests listed above were run independently and passed.
