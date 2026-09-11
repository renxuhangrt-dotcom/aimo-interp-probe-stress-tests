from __future__ import annotations

import csv
import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))

import prepare_dataset
import extract_hidden_states
import train_nested_probe


class DatasetPreparationTests(unittest.TestCase):
    def write_input(self, path: Path, inconsistent: bool = False) -> None:
        fields = ["problem_id", "dataset_id", "original_problem", "model_id", "model_is_robust"]
        rows = [
            {"problem_id": "a", "dataset_id": "s1", "original_problem": "one", "model_id": "m", "model_is_robust": "true"},
            {"problem_id": "a", "dataset_id": "s1", "original_problem": "one", "model_id": "m", "model_is_robust": "false" if inconsistent else "true"},
            {"problem_id": "b", "dataset_id": "s1", "original_problem": "two", "model_id": "m", "model_is_robust": "false"},
            {"problem_id": "c", "dataset_id": "s2", "original_problem": "three", "model_id": "m", "model_is_robust": "true"},
            {"problem_id": "d", "dataset_id": "s2", "original_problem": "four", "model_id": "m", "model_is_robust": "false"},
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_consistent_duplicates_are_collapsed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_input(root / "input.csv")
            audit = prepare_dataset.prepare(root / "input.csv", root / "output.csv", root / "audit.json")
            self.assertEqual(audit["input_rows"], 5)
            self.assertEqual(audit["unique_problem_rows"], 4)
            self.assertEqual(audit["removed_duplicate_rows"], 1)

    def test_inconsistent_duplicates_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_input(root / "input.csv", inconsistent=True)
            with self.assertRaisesRegex(ValueError, "inconsistent duplicate"):
                prepare_dataset.prepare(root / "input.csv", root / "output.csv", root / "audit.json")


class GroupedValidationTests(unittest.TestCase):
    def test_split_has_no_group_overlap(self) -> None:
        groups = np.asarray([f"g{i}" for i in range(20) for _ in range(2)])
        y = np.asarray([i % 2 for i in range(20) for _ in range(2)], dtype=np.int8)
        indices = np.arange(len(y))
        for train, test in train_nested_probe.split_indices(indices, y, groups, 4, 42):
            self.assertFalse(set(groups[train]).intersection(groups[test]))

    def test_nested_oof_finds_training_only_signal(self) -> None:
        rng = np.random.default_rng(7)
        count = 80
        y = np.asarray([index % 2 for index in range(count)], dtype=np.int8)
        groups = np.asarray([f"p{index}" for index in range(count)])
        signal = np.column_stack([y * 8.0 - 4.0, rng.normal(0, 0.1, count)]).astype(np.float32)
        noise = rng.normal(size=(count, 2)).astype(np.float32)
        margins, records = train_nested_probe.nested_oof(
            {0: signal, 1: noise}, y, groups, [0, 1], [0.1, 1.0], 4, 3, [42]
        )
        self.assertGreater(train_nested_probe.balanced_accuracy_score(y, margins >= 0.0), 0.95)
        self.assertTrue(all(record["selected_layer"] == 0 for record in records))


class ExtractionConsolidationTests(unittest.TestCase):
    def test_consolidation_requires_and_preserves_every_problem(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "problems.csv"
            rows = [
                {"problem_id": "p0", "dataset_id": "a", "original_problem": "zero", "model_is_robust": "false"},
                {"problem_id": "p1", "dataset_id": "b", "original_problem": "one", "model_is_robust": "true"},
            ]
            with dataset.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            prompt = root / "prompt.txt"
            prompt.write_text("system", encoding="utf-8")
            cache = root / "cache"
            cache.mkdir()
            for index, row in enumerate(rows):
                digest = extract_hidden_states.fingerprint("model", "system", row["original_problem"], 16)
                path = extract_hidden_states.cache_path(cache, row, digest)
                np.savez_compressed(
                    path,
                    hidden_states=np.full((3, 2), index, dtype=np.float16),
                    fingerprint=np.asarray(digest),
                    token_count=np.asarray(index + 2, dtype=np.int32),
                )
            output = root / "internals"
            extract_hidden_states.consolidate(
                argparse.Namespace(
                    dataset=dataset,
                    system_prompt=prompt,
                    model_id="model",
                    cache_dir=cache,
                    max_length=16,
                    output_dir=output,
                )
            )
            self.assertEqual(np.load(output / "layer_001.npy").shape, (2, 2))
            self.assertTrue((output / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
