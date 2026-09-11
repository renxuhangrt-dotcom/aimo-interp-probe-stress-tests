from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "kaggle_e11_rrb_multiview.py"
NOTEBOOK = ROOT / "aimo_e11a_rrb_multiview.ipynb"

spec = importlib.util.spec_from_file_location("e11", SOURCE)
e11 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = e11
spec.loader.exec_module(e11)


class E11Tests(unittest.TestCase):
    def test_frozen_v6_oof_control(self) -> None:
        old_repository = e11.REPOSITORY_DIR
        e11.REPOSITORY_DIR = ROOT.parents[1] / "work" / "official" / "baselines"
        rows, _ = e11.prepare_official_rows()
        fractions = e11.load_frozen_v6_oof_control(rows)
        labels = np.asarray([bool(row["model_is_robust"]) for row in rows])
        self.assertAlmostEqual(
            e11.balanced_accuracy(labels, fractions >= 0.5),
            e11.V6_REFERENCE_BA,
            places=15,
        )
        e11.REPOSITORY_DIR = old_repository

    def test_deployed_v6_artifact_replay_path(self) -> None:
        source_zip = ROOT.parents[1] / "outputs" / "aimo-small-v6-fixed-layer-vote-20260910.zip"
        old_temp = e11.TEMP_ROOT
        old_download = e11.urllib.request.urlretrieve
        with tempfile.TemporaryDirectory() as directory:
            e11.TEMP_ROOT = Path(directory)
            e11.urllib.request.urlretrieve = lambda _url, destination: shutil.copy2(
                source_zip, destination
            )
            artifact, audit = e11.load_deployed_v6_artifact()
            self.assertEqual(audit["vote_count"], 225)
            cube = np.zeros((2, len(e11.LAYERS), 4096), dtype=np.float32)
            fractions = e11.deployed_v6_vote_fraction(cube, artifact)
            self.assertEqual(fractions.shape, (2,))
            self.assertTrue(np.all((0.0 <= fractions) & (fractions <= 1.0)))
        e11.TEMP_ROOT = old_temp
        e11.urllib.request.urlretrieve = old_download

    def test_probe_and_label_views_are_disjoint(self) -> None:
        self.assertEqual(
            e11.PROBE_VIEWS,
            ["original", "flatten", "irrelevant", "copied_wrapper"],
        )
        self.assertEqual(
            e11.RRB_VARIANTS,
            ["baseline", "sentence_reversal", "word_reversal", "split_reversal"],
        )
        self.assertFalse(set(e11.PROBE_VIEWS[1:]) & set(e11.RRB_VARIANTS[1:]))

    def test_rrb_transformations_match_frozen_examples_and_are_involutions(self) -> None:
        text = "First  sentence. Second line."
        self.assertEqual(e11.sentence_reversal(text), " Second line.First  sentence.")
        self.assertEqual(e11.word_reversal("one  two\tthree"), "three  two\tone")
        self.assertEqual(e11.split_reversal("ab  c"), "ba  c")
        for function in (e11.sentence_reversal, e11.word_reversal, e11.split_reversal):
            self.assertEqual(function(function(text)), text)

    def test_probe_views_are_deterministic_and_retain_problem(self) -> None:
        problem = "Find x.\nThen report it."
        self.assertEqual(e11.probe_view(problem, "original"), problem)
        self.assertEqual(e11.probe_view(problem, "flatten"), "Find x. Then report it.")
        self.assertIn(problem, e11.probe_view(problem, "irrelevant"))
        self.assertIn(problem, e11.probe_view(problem, "copied_wrapper"))
        self.assertEqual(
            e11.probe_view(problem, "irrelevant"),
            e11.probe_view(problem, "irrelevant"),
        )

    def test_multiview_features_have_frozen_shape_and_order(self) -> None:
        rng = np.random.default_rng(7)
        cube = rng.normal(size=(3, 4, 9, 11)).astype(np.float32)
        features, names = e11.make_multiview_features(cube)
        self.assertEqual(features.shape, (3, 18))
        self.assertEqual(len(names), 18)
        self.assertEqual(names[0], "flatten_cosine_mean")
        self.assertEqual(names[-1], "copied_wrapper_relative_l2_max")
        identical = np.repeat(cube[:, :1], 4, axis=1)
        zeros, _ = e11.make_multiview_features(identical)
        self.assertTrue(np.allclose(zeros, 0.0, atol=1e-6))

    def test_answer_extraction(self) -> None:
        self.assertEqual(e11.extract_aime_answer(r"work... \\boxed{007}"), 7)
        self.assertEqual(e11.extract_aime_answer(r"answer is \\boxed{\frac{12}{3}}"), 4)
        self.assertEqual(e11.extract_aime_answer("Therefore, final answer: 204"), 204)
        self.assertIsNone(e11.extract_aime_answer("No numeric conclusion."))
        self.assertIsNone(e11.extract_aime_answer("final answer: 1000"))

    def test_prediction_freeze_rejects_changes(self) -> None:
        old_root = e11.RESULT_DIR
        old_prediction = e11.PREDICTION_FREEZE
        old_audit = e11.PREDICTION_FREEZE_AUDIT
        old_generation = e11.GENERATION_CACHE
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            e11.RESULT_DIR = base / "result"
            e11.PREDICTION_FREEZE = e11.RESULT_DIR / "predictions.csv"
            e11.PREDICTION_FREEZE_AUDIT = e11.RESULT_DIR / "freeze.json"
            e11.GENERATION_CACHE = base / "generation"
            rows = [
                {
                    "problem_id": "p1",
                    "candidate_probability": 0.25,
                    "candidate_prediction": False,
                    "v6_vote_fraction": 0.6,
                    "v6_prediction": True,
                }
            ]
            audit = e11.write_prediction_freeze(rows)
            self.assertEqual(audit["row_count"], 1)
            changed = [{**rows[0], "candidate_probability": 0.75}]
            with self.assertRaisesRegex(RuntimeError, "changed"):
                e11.write_prediction_freeze(changed)
        e11.RESULT_DIR = old_root
        e11.PREDICTION_FREEZE = old_prediction
        e11.PREDICTION_FREEZE_AUDIT = old_audit
        e11.GENERATION_CACHE = old_generation

    def test_notebook_is_valid_and_all_code_cells_compile(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertTrue(notebook["metadata"]["kaggle"]["isGpuEnabled"])
        self.assertTrue(notebook["metadata"]["kaggle"]["isInternetEnabled"])
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        compile(code, str(NOTEBOOK), "exec")
        self.assertIn(e11.MODEL_REVISION, code)
        self.assertNotIn("renxuhang2020", code.lower())


if __name__ == "__main__":
    unittest.main()
