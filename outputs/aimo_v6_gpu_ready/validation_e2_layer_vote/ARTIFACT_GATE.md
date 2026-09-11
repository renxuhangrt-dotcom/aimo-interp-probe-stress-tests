# AIMO Probe Artifact Gate Report

Status: **PASS**

## Evidence

- Model: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`
- Schema/strategy: `3` / `fixed_multilayer_majority_vote`
- Layers: `[4, 8, 12, 16, 20, 24, 28, 32, 36]`
- SHA-256: `7A4DF77D031530DC4998CBC4B6FFF2B8B717F99C7FA04E2646847C6A5BE3672B`
- Selected layer: `None`
- Group counts: `{"NONE": 25}`
- Real-label fold scores at the selected layer: none
- Selected-layer aggregate score: `0.7047579757975797`
- Best single-probe score: `None`
- Best-to-aggregate gap: `None`

## Failures

- None

## Warnings

- None

## Decision

This artifact must not be packaged as V6 unless the status is PASS. Randomized-label
control groups may remain in the research artifact, but only `control_task=NONE`
groups are allowed to vote at inference.
