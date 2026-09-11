# AIMO Small Track Experiment Log

Last updated: 2026-09-10 (Asia/Shanghai)

## Fixed objective and decision rule

- Prize objective: Small Models Track podium as a stretch goal; technical-report prize as the primary realistic route.
- Current visible podium line: 0.68 (13/19); a clean pass requires at least 0.74 (14/19).
- Do not tune on leaderboard score alone.
- A new learned candidate may be submitted only when all local gates pass:
  - grouped balanced accuracy >= 0.65;
  - bootstrap 95% lower bound > 0.55;
  - balanced accuracy >= 0.60 in both source-holdout directions;
  - deterministic output, coverage 1.0, zero invalid predictions;
  - prediction logic and threshold frozen before reading the hidden score.

## Submission ledger

| Submission | ID | Hypothesis | Hidden score | Decision |
|---|---:|---|---:|---|
| `test-baseline-small.zip` | 919560 / 919626 | Always predict non-robust | 0.47 | Establishes the 9/19 constant-negative baseline |
| `aimo-small-v1-uncertainty.zip` | 919680 | Official uncertainty package | 0.00 | Invalid execution path; do not interpret scientifically |
| `aimo-small-v2-safe-uncertainty.zip` | 919695 | Safe uncertainty wrapper | 0.42 | Worse than the constant-negative baseline |
| `aimo-small-v3-answer-stability.zip` | 919725 | Strict generated-answer agreement | 0.47 | No gain over the constant-negative baseline |
| `aimo-small-v4-majority-stability.zip` | 919730 | Majority answer stability | 0.53 | +1 correct case over the constant-negative baseline |
| `aimo-small-v5-semantic-stability.zip` | 919754 | More robust semantic answer normalization | 0.53 | Predictions unchanged from V4; stop behavioural tweaking |
| `aimo-small-probe-control.zip` | 919900 | Layer-31 white-box representation probe ensemble | 0.47 | Valid execution, but all 19 predictions were `false`; equivalent to the constant-negative baseline |

## Probe-control provenance

- Track: Small Models Track (`small.txt` present at ZIP root).
- Model artifact: schema v2, `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`, layer 31, eight probe groups.
- Artifact-reported selection metric: validation balanced accuracy.
- Artifact-reported best individual probe: 0.7375 validation balanced accuracy.
- Inference optimization: cache the tokenizer/model process-wide so the 8B checkpoint is loaded once, rather than once per problem.
- Package SHA-256: `0814F3AE770996423C0A7BE1CC75CABD392AF36BB2CD7834AC11D7F0F9133E8C`.

## Probe-control postmortem

- Scoring result: accuracy `0.47368421052631576`, coverage `1.0`, invalid predictions `0`.
- Prediction result: 19/19 cases were predicted `false`; the submission did execute rather than fail ingestion.
- The artifact's top-level selected-layer validation score is only `0.5059375`; the advertised `0.7375` is one best fold/seed result and is not the ensemble's validated performance.
- The submitted scorer averaged all eight groups, including four `RANDOMIZATION` control-task groups. Randomized-label probes are controls, not inference ensemble members, so this was an implementation error.
- Even after filtering out the randomized controls, three of four real-label folds report chance-level validation at layer 31. Therefore a filter-only resubmission is not justified; the representation pipeline must be retrained and validated before V6.

## Resource status

- Local NVIDIA GPU: none.
- Local compute budget: zero.
- Time budget: approximately 20 hours/week.
- Compute-support draft: `compute_support_application_draft.md`.
- Requested allocation: 2x A100 40GB, 96 total GPU-hours, restartable during the competition.
- Official compute-support form submitted successfully on 2026-09-09; decision email: `renxuhangrt@gmail.com`.
- Codabench quota after probe-control submission: 9/10 daily, 9/100 total.

## Next experiments after compute access

1. Reproduce hidden-state extraction on all 137 unique public problems.
2. Compare single best probe, fold/seed ensemble, and regularized layer ensemble using nested grouped validation.
3. Add uncertainty features only as out-of-fold residual features.
4. Run leave-one-source-out validation and confidence intervals.
5. Freeze the first candidate that passes all gates; preserve the remaining leaderboard submissions for confirmatory tests.

## V6 preflight gate implemented

- Artifact audit tool: `aimo_probe_gate/audit_probe_artifact.py`.
- The submitted control artifact is automatically rejected (`FAIL`): selected-layer aggregate `0.5059375`, missing grouped bootstrap/source-holdout evidence, and missing nested problem-grouped selection metadata.
- Production inference now permits only `control_task=NONE` groups, refuses ambiguous group metadata, and raises on a missing artifact instead of silently predicting all `false`.
- Inference emits compact layer/group/positive-count/margin diagnostics for observable Codabench runs.
- Verification: 7/7 trained-probe/artifact-gate safety tests and 8/8 ingestion/track regression tests passed.
- No V6 archive was produced or submitted.

## GPU-ready grouped-validation workflow frozen

- Official baseline was cloned and pinned at commit `7e8839966750059b6b1d247a12ab56552c79b342`.
- Audit found row-wise classification folds in the official script even though the aggregate data contains repeated `problem_id` rows; this can leak the same problem across train and validation.
- `math-robust-agg.csv` was collapsed from 141 rows to 137 unique, label-consistent problems. Canonical-content SHA-256: `3FC14BFC02F7C1D0D79BB917491C7710F6A798150416B157C6C34205CEC9E067`.
- The frozen workflow uses resumable two-GPU extraction, nested `problem_id`-grouped layer/regularization selection, two source-holdout directions, a shuffled-label control, and a 5,000-resample bootstrap.
- A fourth gate was added: the real-label OOF score must exceed the shuffled-label control by at least `0.05`.
- Five pipeline tests pass, covering duplicate handling, inconsistent-data rejection, cache consolidation, grouped split isolation, and nested selection on a synthetic signal.
- No hidden-state extraction was attempted locally because CUDA is unavailable; no V6 artifact or archive was produced.

## Kaggle T4 x2 extraction edition prepared and corrected

- A self-contained Kaggle notebook was generated for exact-weight FP16 extraction: `aimo_v6_kaggle/aimo_v6_kaggle_v2.ipynb`.
- The notebook loads one 8B model across both T4 devices with `device_map="balanced"`; it does not launch two model replicas and does not quantize weights.
- Fail-closed preflight rejects single GPU, P100, CPU/disk offload, upstream-data drift, invalid dimensions, incomplete coverage, and non-finite activations.
- Each problem is atomically cached under `/kaggle/working`; valid cache files from an attached previous notebook output are imported automatically.
- The final downloadable `aimo_v6_internals.zip` contains metadata, all 37 layer matrices, data audit, progress, device map, and manifest. Model weights are excluded.
- Version 1 failed after 22.7 seconds before model loading because raw CSV byte hashes differed across Windows and Linux newline conventions; the dataset rows and counts were otherwise identical.
- Version 2 canonicalizes line endings and gates on logical content hash `3FC14BFC02F7C1D0D79BB917491C7710F6A798150416B157C6C34205CEC9E067` while retaining the pinned Git commit.
- Corrected notebook SHA-256: `D2024085EE929C3D7F30FC62E0B96F960268578F0F0F70A0369086088E517400`.
- Verification: 6/6 Kaggle-edition tests passed; notebook JSON parsed, its code cell compiled, and embedded-secret scan passed.
- GPU runtime remains untested until queued on Kaggle; still no V6 artifact/archive was produced.

## Kaggle extraction completed; frozen validation failed one gate

- `aimo_v6_internals.zip` SHA-256: `FC49D7E0EB60A02FF6E17F551081378C86A2FE3A2B8C7A640E8F5E82FBD5B47A`.
- Extraction completed 137/137 problems on 2 × Tesla T4 with unquantized FP16 weights, 37 layers, 4096 dimensions, no CPU/disk offload, and a maximum prompt length of only 286 tokens.
- Independent integrity scan confirmed all 37 consolidated float32 matrices have shape 137 × 4096 and contain only finite values.
- Frozen nested grouped OOF balanced accuracy: `0.657178`; bootstrap 95% lower bound: `0.565411`; both pass.
- Shuffled-label control: `0.516364`; real-control advantage: `0.140814`; passes.
- Source holdouts: `0.563333` for test-reference → train-reference and `0.661863` for the reverse direction. The first direction fails the preregistered `0.60` minimum.
- Layer selection is unstable across outer folds (eight different layers selected); 20/25 folds choose strong regularization `C=0.001`.
- Decision: overall gate `FAIL`; no trained artifact or V6 archive was produced.

## E2 fixed multi-layer vote passed; V6 packaged

- E2 was preregistered before execution as a fixed nine-layer majority vote over layers `4, 8, 12, 16, 20, 24, 28, 32, 36`, with `C=0.001`, five grouped folds, and seeds `42..46`.
- Grouped OOF balanced accuracy: `0.704758`; bootstrap 95% lower bound: `0.616944`.
- Source holdouts: `0.675000` for test-reference → train-reference and `0.707317` for the reverse direction.
- Shuffled-label control: `0.585809`; real-control advantage: `0.118949`.
- All four frozen gates passed. Because E1 source-holdout results had already been observed before E2 was designed, E2 remains sequential exploratory evidence rather than an untouched confirmatory experiment.
- The schema-v3 deployment artifact contains 25 real-label fold groups × 9 fixed layers = 225 equal binary votes. Every held-out fold and the aggregate OOF result were replayed through the exact deployment scorer with no discrepancy.
- Artifact gate: `PASS`; artifact SHA-256: `7A4DF77D031530DC4998CBC4B6FFF2B8B717F99C7FA04E2646847C6A5BE3672B`.
- Submission archive: `aimo-small-v6-fixed-layer-vote-20260910.zip`; SHA-256: `C757708FCD5F079AF7A90EC30E91D45E5E9542FDBA32C125A862ACC68D933F3E`; size `4,225,167` bytes.
- ZIP root contains only `solution.py`, `probe_inference.py`, blank `small.txt`, and `probe_artifacts/probe_artifact.pkl`; an extraction-and-replay audit passed.
- Verification: 15/15 ingestion/track/inference tests, 5/5 artifact-gate tests, and 1/1 E2 synthetic-signal test passed. The complete test discovery additionally requires optional local PyTorch dependencies unavailable in this CPU audit environment.
- V6 was packaged locally but not uploaded to Codabench.

## V6 private Small Track result

- Submission archive: `aimo-small-v6-fixed-layer-vote-20260910.zip`.
- Private score: `0.631578947368421` = 12/19 correct; coverage `1.0`; invalid predictions `0`.
- The deployment produced 11 positive and 8 negative predictions. Combining
  this with the 9/19 all-negative baseline implies TP=7, TN=5, FP=4, FN=3 and
  private balanced accuracy approximately `0.627778`.
- This improves by 2 correct cases over V4/V5 (10/19) and by 3 over the
  all-negative/probe-control baseline (9/19).
- The local grouped OOF score was `0.704758`; the private difference is
  plausible for a 19-case test, but the private result is now frozen and will
  not be used for per-case or threshold tuning.

## E3 late-layer vote rejected

- E3 was preregistered as a three-layer `[28, 32, 36]` equal vote with the same
  grouped folds, seeds, `C=0.001`, source holdouts, and shuffled-label control.
- Grouped OOF balanced accuracy fell to `0.680968`; bootstrap lower bound fell
  to `0.595810`.
- Source holdouts improved to `0.686667` and `0.719512` (mean `0.703089`), but
  randomized-control advantage was only `0.080308`.
- OOF disagreement versus E2 was `0.080292`, so the candidate was materially
  different but inferior on the primary and control gates.
- Decision: `FAIL`; no V7 artifact or archive was produced.

## Main Track transfer audit prepared

- The current official evaluation documentation lists four cached checkpoints:
  GPT-OSS 120B, OLMo 3 7B, DeepSeek-R1-Qwen3 8B, and Gemma 3 27B.
- Existing labeled probe training data cover only the `qwen3-8b:low` model.
  Treating the DeepSeek representation score as a model-independent problem
  proxy is therefore a hypothesis, not yet a validated Main Track method.
- The eight unique problems in the public multi-model sample are absent from
  the 137-problem activation cache. The local audit fails closed instead of
  substituting mismatched rows.
- An eight-problem, label-blind Kaggle extraction notebook was prepared as
  `aimo_v6_kaggle/aimo_public_transfer_audit_v2.ipynb`. Its output will support a
  frozen cross-model transfer audit before any Main archive is built.
- The first public-transfer notebook incorrectly preregistered seven unique
  problems and failed closed before model loading. Inspection confirmed the
  frozen 28-row parquet contains eight unique problem texts; v2 corrects only
  that cardinality assertion.

## Frozen V6 cross-model transfer audit: NO-GO

- Public-transfer internals ZIP SHA-256:
  `A0AE165BCA8947557308B23ED0E75C6AB855A5B60A5FFFE978C2AAD4C555EE0E`.
- Integrity passed: 8/8 unique public problems, 37 layers, each matrix
  `8 × 4096` float32, all finite; extraction used exact FP16 DeepSeek weights
  across 2 × Tesla T4 and did not use labels.
- The frozen V6 artifact predicted robust for all 24 public model–problem
  cases. Accuracy was `0.375`, balanced accuracy `0.500`, versus `0.625`
  accuracy for always-False.
- Positive-vote fractions ranged from `0.768889` to `1.000000`, with median
  `0.977778`; this is a strong domain shift, not a few borderline 0.5 votes.
- Excluding the single qwen3-8b public case produced `0.391304` accuracy and
  `0.500` balanced accuracy, still below the `0.608696` always-False baseline.
- Decision: direct cross-model reuse of V6 is `NO-GO`; no Main Track archive
  was produced. A Main candidate must be model-aware or independently
  calibrated across models.

## E4 soft-probability layer ensemble rejected

- E4 was preregistered before execution as the same nine layers, `C=0.001`,
  grouped folds, and seeds as E2, replacing only each probe's hard binary vote
  with its sigmoid decision probability. Probabilities were averaged at a
  fixed `0.5` threshold; no layer weights or threshold search were allowed.
- Grouped OOF balanced accuracy was unchanged at `0.704758`; bootstrap lower
  bound was unchanged at `0.616944`.
- The OOF predictions had exactly `0.000000` disagreement versus E2, showing
  that soft aggregation did not change the decision boundary on any of the
  137 public problems.
- Source holdouts were `0.675000` and `0.686253` (mean `0.680626`), and the
  shuffled-label advantage was `0.136826`.
- Decision: `FAIL` on the primary score, strict bootstrap improvement, mean
  source-holdout, and minimum-disagreement gates. No V7 artifact or archive
  was produced.

## E5 nested consensus calibration passed; V7 packaged

- E5 was preregistered before execution. It retained V6's nine fixed layers,
  balanced logistic probes, `C=0.001`, grouped folds, and seeds. Within each
  outer training partition, four grouped inner folds selected a required
  positive layer-vote count only from `[5, 6, 7, 8]`.
- Grouped OOF balanced accuracy improved from V6's `0.704758` to `0.710671`;
  bootstrap lower bound improved from `0.616944` to `0.624220`.
- Source holdouts were `0.700000` and `0.710643` (mean `0.705322`). The
  randomized-label advantage was `0.159791`.
- OOF disagreement versus V6/E2 was `0.036496` (5/137), satisfying the
  preregistered material-difference gate. Predicted positives decreased from
  69 to 64.
- All E5 promotion gates passed. A schema-v4 artifact was exported with 25
  outer groups × 9 fixed layers; each group retains its own inner-calibrated
  required vote count. Exact deployment replay reproduced `0.710671` on all
  137 OOF problems.
- Artifact SHA-256:
  `D1F86118BD6299E95BD6EE2C245473E3A7D493E8C5A63E7D9EE6202FF7E8B300`.
- Submission archive: `aimo-small-v7-nested-consensus-20260910.zip`; SHA-256:
  `24D7D86993ED0C556A63011E4906D3A759F018CEFE4C04C4A2AE952F77DA3CF9`;
  size `4,227,503` bytes.
- ZIP extraction, schema-v4 artifact gate, Small marker contract, exact
  deployment replay, artifact discovery, and solution entry-point checks
  passed. Relevant local tests: 26/26.
- Decision: V7 is ready for one independent Small Track confirmation. Its
  hidden result must not be used for per-case tuning.

## V7 private Small Track result: rejected

- Codabench result: accuracy `0.47368421052631576` = 9/19, coverage `1.0`,
  invalid predictions `0`.
- Scoring-result SHA-256:
  `B83BCF5BF67684262271B12303C414BE72A077CBD908A61252BA1565BA519050`.
- Prediction-result SHA-256:
  `D0A815229F4C6AA888A86D8459A8CA8348A3F422CDB77E64534E4E9E9A76F7EC`.
- V7 produced 8 positive predictions versus V6's 11. Exactly three hidden
  decisions changed, all monotonically from positive to negative. The score
  declined by exactly three correct cases, so all three stricter-consensus
  changes broke previously correct V6 positives.
- With the hidden class counts inferred from the frozen all-negative baseline,
  V7 has TP=4, TN=5, FP=4, FN=6 and balanced accuracy about `0.477778`, versus
  V6's TP=7, TN=5, FP=4, FN=3 and balanced accuracy about `0.627778`.
- Runtime, coverage, validity, artifact replay, and monotone prediction changes
  rule out an ingestion or packaging failure. This is a calibration-transfer
  failure on a 19-case hidden sample.
- Decision: V7 is a negative result. V6 remains the Small Track champion at
  12/19. Hidden IDs and inferred outcomes are frozen and may not be used for
  case-level or threshold tuning.
- Workflow correction: future promotion requires a positive 95% lower bound
  for the paired problem-level bootstrap difference versus V6, not merely a
  slightly better standalone score/lower bound.

## E6 adjacent-layer representation change rejected

- E6 was preregistered as nine fixed adjacent-layer deltas (`4-0` through
  `36-32`), equal hard voting, `C=0.001`, the established grouped folds and
  seeds, two source holdouts, and a shuffled-label control.
- Grouped OOF balanced accuracy was `0.713696`, only `0.008938` above V6's
  `0.704758`, below the required `0.02` effect size.
- Candidate bootstrap lower bound was `0.630798`, but the paired bootstrap
  difference interval versus V6 was `[-0.014151, 0.007345, 0.042868]`; its
  lower bound crossed zero.
- It disagreed with V6 on only `0.014599` of public problems and corrected one
  V6 error while breaking one V6 correct prediction. Source holdouts were
  `0.675000` and `0.661863` (mean `0.668431`).
- Randomized-control advantage was `0.150715`.
- Decision: `FAIL`; no artifact or archive was produced. The improved paired
  gate successfully rejected a candidate whose standalone score looked
  superficially better but whose practical decisions were almost unchanged.

## E7 problem-token mean representation rejected

- Kaggle problem-view archive SHA-256:
  `FFBD335A4C50040ECAA0DC06E51E8B1054DA296BBDA18DAC375AE07F8B668A2D`;
  size `97,555,614` bytes.
- Integrity passed: 137/137 problems, exact FP16 DeepSeek weights on 2 × Tesla
  T4, 37 layers × 4096 dimensions for both label-blind views, all finite.
  Complete problem spans covered 9–227 tokens. Dataset and official-commit
  gates matched V6.
- E7 evaluated only the preregistered `problem_mean` view. The separately
  extracted `problem_last` view remains unevaluated and may be used only after
  a new preregistration.
- Grouped OOF balanced accuracy was `0.718647`, an improvement of only
  `0.013889` over V6, below the required `0.02`. Ordinary accuracy was
  `0.664234`, one additional correct public problem versus V6.
- Candidate bootstrap lower bound was `0.639071`, but the paired bootstrap
  difference interval versus V6 was `[-0.067573, 0.013055, 0.094004]`; the
  lower bound crossed zero substantially.
- E7 disagreed with V6 on 17/137 problems (`0.124088`), correcting nine V6
  errors and breaking eight V6 correct predictions. Its confusion counts were
  TN=61, FP=40, FN=6, TP=30.
- Source holdouts were strongly asymmetric: `0.611667` and `0.756098` (mean
  `0.683882`). The first direction and mean failed their frozen gates.
- Randomized-label advantage was strong at `0.198295`, indicating a real public
  signal, but not a stable transferable improvement over V6.
- Decision: `FAIL`; no artifact or submission archive was produced. V6 remains
  the Small Track champion.

## E8 final problem-token representation rejected

- E8 was preregistered before the first evaluation of the previously extracted
  label-blind `problem_last` view. It used V6's nine layers, fixed `C=0.001`,
  grouped folds/seeds, equal voting, source holdouts, shuffled-label control,
  and the strengthened paired comparison against V6.
- Grouped OOF balanced accuracy was `0.694857`, which is `-0.009901` versus
  V6. Ordinary accuracy was `0.642336`.
- Candidate bootstrap lower bound was `0.610076`. The paired bootstrap
  difference interval versus V6 was `[-0.071380, -0.009615, 0.050605]`, with a
  negative median and lower bound.
- E8 disagreed with V6 on 10/137 problems (`0.072993`), correcting four V6
  errors and breaking six V6 correct predictions.
- Source holdouts were `0.658333` and `0.661863` (mean `0.660098`). The
  randomized-label advantage was `0.117024`.
- Decision: `FAIL`; no artifact or archive was produced. Both problem-token
  views from the extraction are now evaluated and neither replaces V6.

## E9 PCA10-RBF layer vote rejected

- E9 was preregistered as randomized training-fold PCA with 10 components per
  layer followed by fixed balanced RBF SVC (`C=1`, `gamma=scale`) and equal
  voting across V6's nine layers. No hyperparameter search was permitted.
- Grouped OOF balanced accuracy was `0.712596`, only `0.007838` above V6 and
  below the required `0.02` effect. Ordinary accuracy rose to `0.708029`.
- It disagreed with V6 on 15/137 (`0.109489`), correcting 11 public V6 errors
  and breaking four. However, positive predictions fell from 69 to 56, a
  conservative direction already contradicted by the V7 hidden result.
- Candidate bootstrap lower bound was `0.623326`; the paired bootstrap
  difference interval was `[-0.049604, 0.008844, 0.060748]`, so superiority
  over V6 was not established.
- Source holdouts were asymmetric at `0.628333` and `0.768293` (mean
  `0.698313`); the first direction failed. Randomized-control advantage was
  `0.121012`.
- Decision: `FAIL`; no artifact or archive was produced. Further reuse of the
  same 137 labels for classifier/representation variants is paused pending new
  independent validation evidence.

## Official data audit completed; E10 OOD audit preregistered

- The official 141-row aggregate contains only 137 unique problems; its four
  extra rows are exact label-consistent duplicates. The pinned official
  675-row full file expands the same 137 problems into 2–7 perturbation-family
  rows. Problem-ID and normalized-text intersections are both 137/137.
- Therefore the official filtered/full/aggregate augmentation variants are not
  independent supervised validation sets. Treating their expanded rows as
  independent would reintroduce problem leakage.
- The frozen rule `max(absolute_accuracy_decay) <= 0` exactly reproduces the
  official binary labels for all 137 unique training problems.
- The official 22-row symbolic `problems-public` data has no robustness labels.
  It cannot validate Small-model classification without new annotations.
- E10 was preregistered against the pinned 558-row, ten-problem, eight-model
  `aimo-interp-challenge-sample-full` snapshot. It selects only direct
  `qwen3-8b:low` rows, derives one label per problem with the validated rule,
  proves zero ID/text overlap, and extracts frozen V6 representations.
- E10 is evaluation-only. Fewer than eight direct Small problems, a one-class
  slice, or failure of the preregistered score-over-constant gate cannot
  authorize V8. V6 remains the frozen Small champion while E10 is pending.

## E10 independent Small-model OOD audit failed

- Returned archive SHA-256:
  `32F005F730C0CEC0A75B3D5B0A8333899099E03CA8D8BD2D0284632FF51BCF69`.
- All provenance gates passed: the archive used the pinned 558-row official
  snapshot, selected 64 direct `qwen3-8b:low` perturbation rows covering ten
  unique problems, reproduced the training-label rule on 137/137 reference
  problems, and had zero training problem-ID or normalized-text overlap.
- The ten OOD labels contained eight negative and two positive cases, so the
  preregistered minimum sample-size and two-class requirements were met.
- Frozen V6 predicted all ten cases positive. Accuracy was `0.200000`,
  balanced accuracy was `0.500000`, and confusion was TN=0, FP=8, FN=0,
  TP=2. Its 95% Wilson interval for accuracy was `[0.056682, 0.509838]`.
- The always-negative baseline scored `0.800000`; V6 was `-0.600000` behind
  the best constant baseline. Positive-vote fractions were uniformly high
  (minimum `0.768889`, median `0.940000`, mean `0.919556`, maximum `1.0`).
- Decision: `FAIL`. The result demonstrates a severe OOD positive-bias or
  representation/classifier shortcut, not a threshold near-miss. Per the
  preregistration, E10 cases and vote fractions will not be used for threshold,
  layer, text-rule, or per-case tuning, and V8 is not produced.
- V6 remains the best submitted Small entry (`12/19`, `0.631579`) but is not
  supported as a generalizing scientific method by E10. Work on repeated
  classifiers over the same 137 labels is halted. The next prize-oriented
  deliverable is the technical report, with V6/E6-E10 as reproducible evidence;
  any new model version requires a genuinely new hypothesis and independent
  validation source.
