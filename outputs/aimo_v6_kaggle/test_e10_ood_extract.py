from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))

import kaggle_extract_e10_ood_small as extractor


class E10OodExtractorTests(unittest.TestCase):
    def test_problem_label_uses_maximum_decay_across_rows(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "model_id": "qwen3-8b:low",
                    "dataset_id": "ood",
                    "problem_id": "a",
                    "original_problem": "Problem A",
                    "permutation_type": "rename",
                    "absolute_accuracy_decay": 0.0,
                },
                {
                    "model_id": "qwen3-8b:low",
                    "dataset_id": "ood",
                    "problem_id": "a",
                    "original_problem": "Problem A",
                    "permutation_type": "domain",
                    "absolute_accuracy_decay": 0.5,
                },
                {
                    "model_id": "qwen3-8b:low",
                    "dataset_id": "ood",
                    "problem_id": "b",
                    "original_problem": "Problem B",
                    "permutation_type": "rename",
                    "absolute_accuracy_decay": 0.0,
                },
                {
                    "model_id": "another-model",
                    "dataset_id": "ood",
                    "problem_id": "c",
                    "original_problem": "Problem C",
                    "permutation_type": "rename",
                    "absolute_accuracy_decay": 1.0,
                },
            ]
        )
        rows, selected = extractor.derive_problem_labels(frame)
        self.assertEqual(3, len(selected))
        self.assertEqual(["a", "b"], [row["problem_id"] for row in rows])
        self.assertFalse(rows[0]["model_is_robust"])
        self.assertTrue(rows[1]["model_is_robust"])
        self.assertEqual("domain|rename", rows[0]["permutation_types"])

    def test_conflicting_problem_text_fails_closed(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "model_id": "qwen3-8b:low",
                    "dataset_id": "ood",
                    "problem_id": "a",
                    "original_problem": "Problem A",
                    "permutation_type": "rename",
                    "absolute_accuracy_decay": 0.0,
                },
                {
                    "model_id": "qwen3-8b:low",
                    "dataset_id": "ood",
                    "problem_id": "a",
                    "original_problem": "Different text",
                    "permutation_type": "domain",
                    "absolute_accuracy_decay": 0.0,
                },
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "conflicting original problem"):
            extractor.derive_problem_labels(frame)

    def test_frozen_rule_reproduces_official_training_labels(self) -> None:
        project = PACKAGE.parents[1]
        official = project / "work" / "official" / "baselines"
        if not (official / "data" / "math-robust-final.csv").is_file():
            self.skipTest("requires a local checkout of the pinned official baseline data")
        with mock.patch.object(extractor, "REPOSITORY_DIR", official):
            audit = extractor.validate_label_rule_against_training_reference()
        self.assertEqual(137, audit["unique_problems"])
        self.assertEqual(137, audit["matching_labels"])

    def test_source_has_no_embedded_secret(self) -> None:
        text = (PACKAGE / "kaggle_extract_e10_ood_small.py").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"hf_[A-Za-z0-9]{20,}", text))
        self.assertNotIn("KAGGLE_KEY", text)


if __name__ == "__main__":
    unittest.main()
