from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))

import train_layer_vote_probe as vote


class FixedLayerVoteTests(unittest.TestCase):
    def test_majority_signal_layers_generalize_without_layer_selection(self) -> None:
        generator = np.random.default_rng(17)
        count = 80
        y = np.asarray([index % 2 for index in range(count)], dtype=np.int8)
        groups = np.asarray([f"p{index}" for index in range(count)])
        signal = y * 6.0 - 3.0
        layers = {
            4: np.column_stack([signal, generator.normal(0, 0.2, count)]).astype(np.float32),
            8: np.column_stack([signal, generator.normal(0, 0.2, count)]).astype(np.float32),
            12: np.column_stack([signal, generator.normal(0, 0.2, count)]).astype(np.float32),
            16: generator.normal(size=(count, 2)).astype(np.float32),
            20: generator.normal(size=(count, 2)).astype(np.float32),
        }
        fractions, records = vote.repeated_oof(layers, y, groups, 4, [42], 0.001)
        self.assertEqual(len(records), 4)
        self.assertGreater(balanced_accuracy_score(y, fractions >= 0.5), 0.95)


if __name__ == "__main__":
    unittest.main()
