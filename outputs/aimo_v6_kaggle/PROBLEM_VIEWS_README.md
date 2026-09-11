# E7 problem-token views: Kaggle instructions

Use `aimo_problem_token_views.ipynb`. It performs label-blind activation
extraction only; it does not train, package, or submit a competition entry.

## Run

1. In Kaggle choose **Create → New Notebook**.
2. Choose **File → Import Notebook** and select
   `aimo_problem_token_views.ipynb`.
3. In **Settings**, select **GPU T4 x2**. Do not select P100 or a single GPU.
4. Enable **Internet**.
5. Choose **Save Version → Save & Run All**.
6. When the version succeeds, open **Output** and download
   `aimo_problem_token_views.zip`.
7. Give that ZIP to Codex for the frozen E7 CPU validation.

The previous extraction took about five minutes on the user's Kaggle T4 x2
session. This run uses the same 137 forward passes and saves two views, so a
reasonable estimate is about 5–10 minutes after the GPU starts, plus queue and
download time. The output is expected to be roughly twice the size of the V6
internals archive.

The notebook refuses the wrong GPU topology, P100, CPU/disk model offload,
dataset drift, incomplete problem-span token coverage, incorrect tensor shapes,
non-finite values, and incomplete extraction. If interrupted, attach this run's
Output to the next run as an Input to reuse valid per-problem caches.

Only `problem_mean` is the preregistered E7 representation. `problem_last` is
saved label-blind for a possible separately preregistered future experiment and
must not be selected by looking at E7 results.

