# AIMO E11a Kaggle validation notebook

This package tests a new Small-track hypothesis without using Codabench as the
validation set.  It does **not** produce a submission ZIP.

## Run

1. Create a private Kaggle notebook and import `aimo_e11a_rrb_multiview.ipynb`.
2. In notebook settings select **GPU T4 x2** and turn **Internet on**.
3. Optionally add a private Kaggle secret named `HF_TOKEN`.  The pinned model
   is public, so the notebook can also run without it.
4. Run **Save Version / Run All**.
5. Download `aimo_e11a_rrb_multiview_results.zip` from the notebook Output tab.

Stage A is roughly 668 short forward passes.  If it fails, the notebook stops
before expensive answer generation.  If it passes, Stage B adds 240 sampled
generations and can take several hours.  The generation cache is checkpointed
after every problem/transformation pair.

Read `E11_PREREGISTRATION.md` before running. A scientific `FAIL` is a valid
result and should not be resubmitted with modified thresholds.
