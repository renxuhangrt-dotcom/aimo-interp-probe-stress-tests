# E11a result analysis

Result archive SHA-256:
`777235F7EFC03AC31165A9EF914E3CAABE921D9AD0849170F7429D843A136137`.

Decision: **FAIL_STAGE_A**. V6 remains the Small-track champion; no V8 was
created and no external RRB robustness label was generated.

## Integrity

- Exact pinned model/data/hardware checks passed.
- 668/668 feature jobs completed.
- The immutable V6 OOF control reproduced balanced accuracy `0.7047579758`.
- The deployed V6 archive passed its ZIP/artifact hashes and exposed the
  expected 25 groups and 225 votes.
- AIME 2024 had 30 rows and zero ID or normalized-text overlap with training.
- All 90 RRB transformation reversibility checks passed.
- Neither E10 labels nor external labels entered Stage A.

## Candidate result

- Balanced accuracy: `0.6273377338` versus required `0.7247579758`.
- Ordinary accuracy: `0.6350364964` (`87/137`).
- Bootstrap 95% lower bound: `0.5344018333` versus required strict lower bound
  above `0.6169437632`.
- Paired improvement lower bound versus V6: `-0.1562518211` versus required
  value above zero.
- Source holdouts: `0.5900000000` and `0.7172949002`; mean `0.6536474501`.
- Shuffled-label control: `0.5936468647`; candidate advantage only
  `0.0336908691`, versus required `0.10`.
- Disagreement with V6: `25/137` (`0.1824817518`), which passed its diversity
  gate but did not translate to better predictions.
- Confusion counts: TN=65, FP=36, FN=14, TP=22. Relative to V6's historical
  TN=61, FP=40, FN=7, TP=29, E11a gained four true negatives but lost seven
  true positives.

## Interpretation

The fixed hidden-state distance summaries are unstable across sources and
barely outperform their shuffled-label control. This is not a threshold
near-miss: tuning C, the final threshold, layer subsets, or individual views
would reuse the same labels after seeing failure and is therefore rejected.

The untouched external AIME/RRB label source remains available for a later,
genuinely new candidate. The next defensible hypothesis is targeted
metacognitive elicitation: change the model prompt so its final-token state is
explicitly asked to represent expected paraphrase robustness, rather than
trying to infer robustness from generic latent distances. It must retain the
same frozen gates and reach Stage A before any RRB labels are generated.
