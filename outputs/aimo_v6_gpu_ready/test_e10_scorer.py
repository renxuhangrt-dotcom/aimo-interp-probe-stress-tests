#!/usr/bin/env python3
"""Regression tests for the frozen E10/V6 scoring path."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load_scorer():
    path = HERE / "score_e10_ood_v6.py"
    spec = importlib.util.spec_from_file_location("score_e10_ood_v6", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCORER = load_scorer()


class E10ScorerTests(unittest.TestCase):
    def test_wilson_interval_contains_observed_accuracy(self) -> None:
        lower, upper = SCORER.wilson_interval(7, 10)
        self.assertLess(lower, 0.7)
        self.assertGreater(upper, 0.7)
        self.assertGreaterEqual(lower, 0.0)
        self.assertLessEqual(upper, 1.0)

    def test_preregistered_gate_states(self) -> None:
        labels = np.asarray([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
        passing_predictions = np.asarray([1, 1, 1, 1, 0, 0, 0, 0, 0, 1], dtype=bool)
        passing = SCORER.evaluate(labels, passing_predictions, passing_predictions.astype(float))
        self.assertEqual(passing["decision"], "PASS")
        self.assertAlmostEqual(passing["accuracy"], 0.9)
        self.assertAlmostEqual(passing["advantage_over_best_constant"], 0.3)

        failing_predictions = np.ones(10, dtype=bool)
        failing = SCORER.evaluate(labels, failing_predictions, failing_predictions.astype(float))
        self.assertEqual(failing["decision"], "FAIL")

        short_labels = labels[:7]
        short_predictions = short_labels.copy()
        inconclusive = SCORER.evaluate(
            short_labels, short_predictions, short_predictions.astype(float)
        )
        self.assertEqual(inconclusive["decision"], "INCONCLUSIVE")

    def test_frozen_v6_public_transfer_replay(self) -> None:
        artifact = SCORER.load_v6_artifact(
            ROOT / "outputs" / "aimo-small-v6-fixed-layer-vote-20260910.zip"
        )
        internals = ROOT / "work" / "public_transfer_a0ae165b" / "internals"
        layers = {
            layer: np.asarray(np.load(internals / f"layer_{layer:03d}.npy"), dtype=np.float32)
            for layer in SCORER.EXPECTED_LAYERS
        }
        predictions, fractions = SCORER.predict(artifact, layers)

        self.assertEqual(predictions.shape, (8,))
        self.assertTrue(predictions.all())
        expected_per_problem = np.asarray(
            [
                0.8622222222222222,
                0.9777777777777777,
                0.8666666666666667,
                0.7688888888888888,
                0.8533333333333334,
                1.0,
                0.9111111111111111,
                0.9866666666666667,
            ]
        )
        np.testing.assert_allclose(
            np.sort(fractions), np.sort(expected_per_problem), rtol=0.0, atol=1e-15
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
