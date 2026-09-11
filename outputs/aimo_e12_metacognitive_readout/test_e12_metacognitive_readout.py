from __future__ import annotations

import importlib.util
import inspect
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "kaggle_e12_metacognitive_readout.py"
NOTEBOOK = ROOT / "aimo_e12_metacognitive_readout.ipynb"

spec = importlib.util.spec_from_file_location("e12", SOURCE)
e12 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = e12
spec.loader.exec_module(e12)


class E12Tests(unittest.TestCase):
    def test_frozen_v6_oof_control(self) -> None:
        old_repository = e12.REPOSITORY_DIR
        e12.REPOSITORY_DIR = ROOT.parents[1] / "work" / "official" / "baselines"
        rows, _ = e12.prepare_official_rows()
        fractions = e12.load_frozen_v6_oof_control(rows)
        labels = np.asarray([bool(row["model_is_robust"]) for row in rows])
        self.assertAlmostEqual(
            e12.balanced_accuracy(labels, fractions >= 0.5),
            e12.V6_REFERENCE_BA,
            places=15,
        )
        e12.REPOSITORY_DIR = old_repository

    def test_deployed_v6_artifact_replay_path(self) -> None:
        source_zip = ROOT.parents[1] / "outputs" / "aimo-small-v6-fixed-layer-vote-20260910.zip"
        old_temp = e12.TEMP_ROOT
        old_download = e12.urllib.request.urlretrieve
        with tempfile.TemporaryDirectory() as directory:
            e12.TEMP_ROOT = Path(directory)
            e12.urllib.request.urlretrieve = lambda _url, destination: shutil.copy2(
                source_zip, destination
            )
            artifact, audit = e12.load_deployed_v6_artifact()
            self.assertEqual(audit["vote_count"], 225)
            cube = np.zeros((2, len(e12.LAYERS), 4096), dtype=np.float32)
            fractions = e12.deployed_v6_vote_fraction(cube, artifact)
            self.assertEqual(fractions.shape, (2,))
            self.assertTrue(np.all((0.0 <= fractions) & (fractions <= 1.0)))
        e12.TEMP_ROOT = old_temp
        e12.urllib.request.urlretrieve = old_download

    def test_diagnostic_views_are_counterbalanced(self) -> None:
        self.assertEqual(
            e12.DIAGNOSTIC_VIEWS,
            [
                "robustness_stable_a",
                "robustness_stable_b",
                "correctness_correct_a",
                "correctness_correct_b",
            ],
        )
        problem = "Find x."
        robust_a = e12.diagnostic_prompt(problem, "robustness_stable_a")
        robust_b = e12.diagnostic_prompt(problem, "robustness_stable_b")
        correct_a = e12.diagnostic_prompt(problem, "correctness_correct_a")
        correct_b = e12.diagnostic_prompt(problem, "correctness_correct_b")
        stable = "The model would give the same final answer"
        correct = "The model would give the mathematically correct final answer"
        self.assertIn("A. " + stable, robust_a)
        self.assertIn("B. " + stable, robust_b)
        self.assertIn("A. " + correct, correct_a)
        self.assertIn("B. " + correct, correct_b)
        for prompt in (robust_a, robust_b, correct_a, correct_b):
            self.assertIn(problem, prompt)
            self.assertTrue(prompt.endswith("Return only A or B."))

    def test_option_tokens_must_be_distinct_single_tokens(self) -> None:
        class GoodTokenizer:
            def encode(self, value, add_special_tokens=False):
                return {" A": [17], " B": [23]}[value]

        class BadTokenizer:
            def encode(self, value, add_special_tokens=False):
                return [1, 2]

        self.assertEqual(e12.option_token_ids(GoodTokenizer()), (17, 23))
        with self.assertRaisesRegex(RuntimeError, "not single tokens"):
            e12.option_token_ids(BadTokenizer())

    def test_metacognitive_features_have_frozen_shape_and_order(self) -> None:
        rng = np.random.default_rng(7)
        cube = rng.normal(size=(3, 4, len(e12.LAYERS))).astype(np.float32)
        features, names = e12.make_metacognitive_features(cube)
        self.assertEqual(features.shape, (3, 18))
        self.assertEqual(len(names), 18)
        self.assertEqual(names[0], "robustness_semantic_margin_layer_4")
        self.assertEqual(names[-1], "robustness_minus_correctness_layer_36")
        expected_robust = cube[:, 0:2, :].mean(axis=1)
        expected_contrast = expected_robust - cube[:, 2:4, :].mean(axis=1)
        self.assertTrue(np.allclose(features[:, :9], expected_robust))
        self.assertTrue(np.allclose(features[:, 9:], expected_contrast))

    def test_semantic_sign_correction_makes_option_orders_comparable(self) -> None:
        cube = np.zeros((2, 4, len(e12.LAYERS)), dtype=np.float32)
        cube[:, 0, :] = 2.0
        cube[:, 1, :] = 2.0
        cube[:, 2, :] = -1.0
        cube[:, 3, :] = -1.0
        diagnostics = e12.option_order_diagnostics(cube)
        self.assertEqual(diagnostics["robustness"]["semantic_sign_agreement"], 1.0)
        self.assertEqual(diagnostics["correctness"]["semantic_sign_agreement"], 1.0)
        self.assertEqual(diagnostics["robustness"]["mean_absolute_margin_difference"], 0.0)
        features, _ = e12.make_metacognitive_features(cube)
        self.assertTrue(np.allclose(features[:, :9], 2.0))
        self.assertTrue(np.allclose(features[:, 9:], 3.0))

    def test_stage_a_has_no_external_inputs(self) -> None:
        parameters = set(inspect.signature(e12.stage_a_validation).parameters)
        self.assertEqual(parameters, {"official_rows", "official_cube", "feature_names"})

    def test_probe_and_label_transformations_are_disjoint(self) -> None:
        self.assertFalse(set(e12.DIAGNOSTIC_VIEWS) & set(e12.RRB_VARIANTS))
        self.assertEqual(
            e12.RRB_VARIANTS,
            ["baseline", "sentence_reversal", "word_reversal", "split_reversal"],
        )

    def test_rrb_transformations_match_frozen_examples_and_are_involutions(self) -> None:
        text = "First  sentence. Second line."
        self.assertEqual(e12.sentence_reversal(text), " Second line.First  sentence.")
        self.assertEqual(e12.word_reversal("one  two\tthree"), "three  two\tone")
        self.assertEqual(e12.split_reversal("ab  c"), "ba  c")
        for function in (e12.sentence_reversal, e12.word_reversal, e12.split_reversal):
            self.assertEqual(function(function(text)), text)

    def test_answer_extraction(self) -> None:
        self.assertEqual(e12.extract_aime_answer(r"work... \\boxed{007}"), 7)
        self.assertEqual(e12.extract_aime_answer(r"answer is \\boxed{\frac{12}{3}}"), 4)
        self.assertEqual(e12.extract_aime_answer("Therefore, final answer: 204"), 204)
        self.assertIsNone(e12.extract_aime_answer("No numeric conclusion."))
        self.assertIsNone(e12.extract_aime_answer("final answer: 1000"))

    def test_prediction_freeze_rejects_changes(self) -> None:
        old_root = e12.RESULT_DIR
        old_prediction = e12.PREDICTION_FREEZE
        old_audit = e12.PREDICTION_FREEZE_AUDIT
        old_generation = e12.GENERATION_CACHE
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            e12.RESULT_DIR = base / "result"
            e12.PREDICTION_FREEZE = e12.RESULT_DIR / "predictions.csv"
            e12.PREDICTION_FREEZE_AUDIT = e12.RESULT_DIR / "freeze.json"
            e12.GENERATION_CACHE = base / "generation"
            rows = [
                {
                    "problem_id": "p1",
                    "candidate_probability": 0.25,
                    "candidate_prediction": False,
                    "v6_vote_fraction": 0.6,
                    "v6_prediction": True,
                }
            ]
            audit = e12.write_prediction_freeze(rows)
            self.assertEqual(audit["row_count"], 1)
            changed = [{**rows[0], "candidate_probability": 0.75}]
            with self.assertRaisesRegex(RuntimeError, "changed"):
                e12.write_prediction_freeze(changed)
        e12.RESULT_DIR = old_root
        e12.PREDICTION_FREEZE = old_prediction
        e12.PREDICTION_FREEZE_AUDIT = old_audit
        e12.GENERATION_CACHE = old_generation

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
        self.assertIn(e12.MODEL_REVISION, code)
        self.assertIn(e12.DIAGNOSTIC_PREFILL, code)
        self.assertNotIn("renxuhang2020", code.lower())


if __name__ == "__main__":
    unittest.main()
