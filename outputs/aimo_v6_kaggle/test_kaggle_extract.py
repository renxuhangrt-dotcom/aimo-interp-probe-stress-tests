from __future__ import annotations

import csv
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))

import kaggle_extract_v6 as extractor


class KaggleExtractorTests(unittest.TestCase):
    def test_cache_validation_is_fingerprint_and_shape_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"problem_id": "p", "original_problem": "x"}
            with mock.patch.object(extractor, "CACHE_DIR", root):
                digest = extractor.fingerprint("system", "x")
                path = extractor.cache_path(row, digest)
                np.savez_compressed(
                    path,
                    hidden_states=np.ones((37, 4096), dtype=np.float16),
                    fingerprint=np.asarray(digest),
                    token_count=np.asarray(4, dtype=np.int32),
                )
                self.assertTrue(extractor.cache_valid(path, digest))
                self.assertFalse(extractor.cache_valid(path, "wrong"))

    def test_model_device_map_rejects_cpu_offload(self) -> None:
        model = mock.Mock(hf_device_map={"embed": 0, "layer": 1, "head": "cpu"})
        with self.assertRaisesRegex(RuntimeError, "CPU/disk offload"):
            extractor.model_devices(model)

    def test_model_device_map_requires_both_gpus(self) -> None:
        model = mock.Mock(hf_device_map={"embed": 0, "layer": 0})
        with self.assertRaisesRegex(RuntimeError, "span both GPUs"):
            extractor.model_devices(model)

    def test_previous_cache_import_is_non_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            source = input_root / "old-run" / "problem_cache"
            destination = root / "working-cache"
            source.mkdir(parents=True)
            destination.mkdir()
            (source / "a.npz").write_bytes(b"old")
            (destination / "a.npz").write_bytes(b"current")
            (source / "b.npz").write_bytes(b"new")
            original_path = Path

            def redirected_path(value):
                if str(value).replace("\\", "/") == "/kaggle/input":
                    return input_root
                return original_path(value)

            with (
                mock.patch.object(extractor, "CACHE_DIR", destination),
                mock.patch.object(extractor, "Path", side_effect=redirected_path),
            ):
                imported = extractor.import_previous_cache()
            self.assertEqual(imported, 1)
            self.assertEqual((destination / "a.npz").read_bytes(), b"current")
            self.assertEqual((destination / "b.npz").read_bytes(), b"new")

    def test_notebook_source_has_no_embedded_secret(self) -> None:
        text = (PACKAGE / "kaggle_extract_v6.py").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"hf_[A-Za-z0-9]{20,}", text))
        self.assertNotIn("KAGGLE_KEY", text)

    def test_official_dataset_reproduces_frozen_hash(self) -> None:
        project = PACKAGE.parents[1]
        official = project / "work" / "official" / "baselines"
        if not (official / "data" / "math-robust-agg.csv").is_file():
            self.skipTest("requires a local checkout of the pinned official baseline data")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(extractor, "REPOSITORY_DIR", official),
                mock.patch.object(extractor, "WORKING_ROOT", root),
                mock.patch.object(extractor, "PREPARED_DATASET", root / "problems.csv"),
                mock.patch.object(extractor, "DATASET_AUDIT", root / "audit.json"),
            ):
                rows, audit = extractor.prepare_dataset()
            self.assertEqual(len(rows), 137)
            self.assertEqual(
                audit["canonical_content_sha256"],
                               "3FC14BFC02F7C1D0D79BB917491C7710F6A798150416B157C6C34205CEC9E067",
            )


if __name__ == "__main__":
    unittest.main()
