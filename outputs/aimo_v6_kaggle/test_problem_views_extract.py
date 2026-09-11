from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))

import kaggle_extract_problem_views as extractor


class FakeTokenizer:
    def __init__(self, offsets: list[tuple[int, int]]):
        self.offsets = offsets

    def __call__(self, prompt: str, **kwargs):
        del prompt, kwargs
        return {
            "input_ids": np.zeros((1, len(self.offsets)), dtype=np.int64),
            "attention_mask": np.ones((1, len(self.offsets)), dtype=np.int64),
            "offset_mapping": np.asarray([self.offsets], dtype=np.int64),
        }


class ProblemViewExtractorTests(unittest.TestCase):
    def test_problem_span_uses_only_overlapping_tokens(self) -> None:
        prompt = "SYS<user>abc def</user><assistant>"
        tokenizer = FakeTokenizer(
            [(0, 0), (0, 3), (3, 9), (9, 12), (12, 16), (16, 23), (23, 34)]
        )
        encoded, positions = extractor.tokenize_with_problem_span(
            tokenizer, prompt, "abc def"
        )
        self.assertEqual([3, 4], positions)
        self.assertNotIn("offset_mapping", encoded)

    def test_incomplete_problem_span_fails_closed(self) -> None:
        prompt = "SYS<user>abc def</user>"
        tokenizer = FakeTokenizer([(0, 9), (9, 12)])
        with self.assertRaisesRegex(RuntimeError, "truncated or incompletely"):
            extractor.tokenize_with_problem_span(tokenizer, prompt, "abc def")

    def test_cache_requires_both_views_and_span_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            digest = "digest"
            np.savez_compressed(
                path,
                problem_mean_hidden_states=np.ones((37, 4096), dtype=np.float16),
                problem_last_hidden_states=np.ones((37, 4096), dtype=np.float16),
                fingerprint=np.asarray(digest),
                token_count=np.asarray(12, dtype=np.int32),
                problem_span_token_count=np.asarray(4, dtype=np.int32),
            )
            self.assertTrue(extractor.cache_valid(path, digest))
            self.assertFalse(extractor.cache_valid(path, "wrong"))

    def test_model_device_map_rejects_cpu_offload(self) -> None:
        model = mock.Mock(hf_device_map={"embed": 0, "layer": 1, "head": "cpu"})
        with self.assertRaisesRegex(RuntimeError, "CPU/disk offload"):
            extractor.model_devices(model)

    def test_source_has_no_embedded_secret(self) -> None:
        text = (PACKAGE / "kaggle_extract_problem_views.py").read_text(encoding="utf-8")
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
            self.assertEqual(137, len(rows))
            self.assertEqual(
                "3FC14BFC02F7C1D0D79BB917491C7710F6A798150416B157C6C34205CEC9E067",
                audit["canonical_content_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
