# AIMO E12 Kaggle validation notebook

E12 tests whether a counterbalanced, explicit metacognitive readout predicts
paraphrase robustness better than V6. It is a fail-closed validation experiment,
not a Codabench submission.

## Run

1. Read `E12_PREREGISTRATION.md`.
2. Create a private Kaggle notebook and import
   `aimo_e12_metacognitive_readout.ipynb`.
3. Select **GPU T4 x2** and turn **Internet on**.
4. Optionally add the private Kaggle secret `HF_TOKEN`; the pinned model is
   public, so it may run without one.
5. Choose **Save Version / Run All**.
6. Download `aimo_e12_metacognitive_readout_results.zip` from **Output**.

Stage A runs 548 short forward passes and should be comparable to the E11a
runtime. If it fails, no external dataset or label generation is touched. If it
passes, Stage B performs another 150 feature passes and then up to 240 sampled
generations, which can take several hours.

Do not rerun a failed result with adjusted prompts, thresholds, layers, or
classifier settings under the E12 name.
