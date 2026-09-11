#!/usr/bin/env python3
"""Check that report claims remain synchronized with frozen result JSONs."""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
VALIDATION = HERE.parent / "aimo_v6_gpu_ready"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class ReportIntegrityTests(unittest.TestCase):
    def test_machine_table_matches_frozen_results(self) -> None:
        with (HERE / "results_summary.csv").open(encoding="utf-8", newline="") as handle:
            rows = {row["experiment"]: row for row in csv.DictReader(handle)}
        directories = {
            "V6": "validation_e2_layer_vote",
            "E6": "validation_e6_layer_delta_vote",
            "E7": "validation_e7_problem_mean",
            "E8": "validation_e8_problem_last",
            "E9": "validation_e9_pca_rbf",
        }
        self.assertEqual(set(rows), set(directories))
        for experiment, directory in directories.items():
            result = read_json(VALIDATION / directory / "validation_result.json")
            metrics = result["metrics"]
            self.assertEqual(
                rows[experiment]["grouped_oof_balanced_accuracy"],
                f'{metrics["grouped_balanced_accuracy"]:.6f}',
            )
            self.assertEqual(
                rows[experiment]["grouped_oof_accuracy"],
                f'{metrics["ordinary_accuracy"]:.6f}',
            )
            self.assertEqual(rows[experiment]["gate_passed"], str(result["gate"]["passed"]))

    def test_reports_contain_core_frozen_claims(self) -> None:
        v6 = read_json(VALIDATION / "validation_e2_layer_vote" / "validation_result.json")
        e10 = read_json(VALIDATION / "validation_e10_ood_small" / "validation_result.json")
        required = [
            f'{v6["metrics"]["grouped_balanced_accuracy"]:.4f}',
            "12/19",
            f'{e10["accuracy"]:.2f}',
            f'{e10["best_constant_accuracy"]:.2f}',
            "predicted every",
        ]
        for filename in ["TECHNICAL_REPORT_DRAFT.md", "TECHNICAL_REPORT_2PAGE_DRAFT.md"]:
            text = (HERE / filename).read_text(encoding="utf-8")
            for claim in required:
                self.assertIn(claim, text, msg=f"{claim!r} missing from {filename}")

    def test_local_report_dependencies_exist(self) -> None:
        for filename in [
            "results_overview.svg",
            "results_overview.png",
            "results_summary.csv",
            "e10_summary.csv",
            "reproducibility_manifest.json",
            "build_two_page_pdf.py",
            "pdf/When_Good_Probes_Fail_Xuhang_Ren_submission.pdf",
            "NEW_MODEL_GATE.md",
            "ROADMAP_ZH.md",
            "SUBMISSION_CHECKLIST.md",
        ]:
            path = HERE / filename
            self.assertTrue(path.is_file(), msg=f"missing {path}")
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
