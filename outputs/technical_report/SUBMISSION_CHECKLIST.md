# Technical-report submission checklist

## Already complete

- [x] Author: `XUHANG REN`; affiliation: `Independent Researcher`.
- [x] Public contact: `renxuhang2020@qq.com`.
- [x] Frozen V6 artifact, archive, and hashes.
- [x] Machine-readable E6–E10 results and preregistrations.
- [x] Public duplicate/leakage audit.
- [x] Independent ten-problem E10 audit with zero ID/text overlap.
- [x] Result table generated from frozen JSON rather than hand transcription.
- [x] Reproducibility manifest and figure source.
- [x] Explicit negative-result and claim-boundary sections.
- [x] New-model gate forbidding V6/E10 feedback tuning.

## Required before public release

- [ ] Confirm the organizer's final page limit/template; current website is the
  operative source, while the earlier proposal describes a 1–2-page report.
- [ ] Create a clean public repository and replace the code URL placeholder.
- [ ] Add a LICENSE and verify compatibility with the official baseline's
  Apache-2.0 license and model license.
- [ ] Run secret, large-file, archive-layout, and fresh-environment tests.
- [ ] Re-run `build_report_assets.py` and freeze the final manifest.
- [ ] Record final Small Track rank/score only after the competition closes;
  do not use it to tune the method.
- [ ] Submit the report and open-source implementation by 2026-11-15 under the
  current official schedule.
- [ ] Be prepared to peer-review up to three other reports during 2026-11-15 to
  2026-11-30 if considered for an award.
