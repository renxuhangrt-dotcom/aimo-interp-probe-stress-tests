#!/usr/bin/env python3
"""Replay E2 through the exact schema-v3 deployment scoring function."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
from sklearn.metrics import balanced_accuracy_score


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_inference_importable() -> None:
    """Stub heavyweight imports when only the NumPy scorer is under test."""
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        torch_stub = ModuleType("torch")
        torch_stub.Tensor = type("Tensor", (), {})
        sys.modules["torch"] = torch_stub
    try:
        import transformers  # noqa: F401
    except ModuleNotFoundError:
        stub = ModuleType("transformers")
        stub.AutoModelForCausalLM = object
        stub.AutoTokenizer = object
        sys.modules["transformers"] = stub


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--internals", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--training-module", type=Path, required=True)
    args = parser.parse_args()

    make_inference_importable()
    inference = load_module("v6_deployed_probe_inference", args.inference)
    training = load_module("v6_nested_training", args.training_module)
    artifact = inference.load_pickle_artifact(args.artifact)
    layers = inference._required_layer_indices(artifact)
    rows = training.read_metadata(args.internals / "metadata.csv")
    labels = training.boolean_labels(rows)
    problem_ids = np.asarray([row["problem_id"] for row in rows])
    matrices = training.load_layers(args.internals, layers)
    all_indices = np.arange(len(rows))
    oof_positive_fractions: list[list[float]] = [[] for _ in rows]

    eligible = inference._eligible_inference_groups(artifact.data)
    if len(eligible) != 25:
        raise RuntimeError(f"expected 25 eligible groups, got {len(eligible)}")
    for group in eligible:
        seed = int(group["seed"])
        fold_index = int(group["fold_index"])
        splits = list(training.split_indices(all_indices, labels, problem_ids, 5, seed))
        _, test_indices = splits[fold_index]
        one_group_data = dict(artifact.data)
        one_group_data["groups"] = [group]
        one_group_artifact = inference.ProbeArtifact(
            model_id=artifact.model_id,
            system_prompt=artifact.system_prompt,
            data=one_group_data,
            kind=artifact.kind,
        )
        fold_predictions = []
        for row_index in test_indices:
            vectors = {
                layer: np.asarray(matrices[layer][row_index], dtype=np.float32)
                for layer in layers
            }
            fraction = inference.multilayer_positive_vote_fraction(
                vectors, one_group_artifact
            )
            fold_predictions.append(fraction >= 0.5)
            oof_positive_fractions[int(row_index)].append(fraction)
        score = float(
            balanced_accuracy_score(labels[test_indices], fold_predictions)
        )
        expected = float(group["validation_vote_balanced_accuracy"])
        if abs(score - expected) > 1e-12:
            raise RuntimeError(
                f"deployed replay mismatch seed={seed} fold={fold_index}: "
                f"{score} != {expected}"
            )

    if any(len(fractions) != 5 for fractions in oof_positive_fractions):
        raise RuntimeError("each problem must receive one held-out group from each seed")
    oof_predictions = np.asarray(
        [np.mean(fractions) >= 0.5 for fractions in oof_positive_fractions]
    )
    replay_score = float(balanced_accuracy_score(labels, oof_predictions))
    expected_replay = float(artifact.data["validation_replay"]["balanced_accuracy"])
    if abs(replay_score - expected_replay) > 1e-12:
        raise RuntimeError(
            f"aggregate deployed replay mismatch: {replay_score} != {expected_replay}"
        )

    first_vectors = {
        layer: np.asarray(matrices[layer][0], dtype=np.float32) for layer in layers
    }
    full_fraction = inference.multilayer_positive_vote_fraction(first_vectors, artifact)
    summary = {
        "passed": True,
        "rows": len(rows),
        "eligible_groups": len(eligible),
        "layers": layers,
        "total_deployment_votes": len(eligible) * len(layers),
        "oof_replay_balanced_accuracy": replay_score,
        "first_case_full_ensemble_positive_vote_fraction": full_fraction,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
