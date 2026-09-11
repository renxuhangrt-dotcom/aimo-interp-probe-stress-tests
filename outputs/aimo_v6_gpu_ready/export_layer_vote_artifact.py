#!/usr/bin/env python3
"""Export the E2 fixed-layer fold ensemble as a plain-data schema-v3 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

import train_nested_probe as base


EXPECTED_EXPERIMENT = "E2_fixed_multilayer_vote"
EXPECTED_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32, 36]
EXPECTED_C = 0.001
EXPECTED_SEEDS = [42, 43, 44, 45, 46]
EXPECTED_FOLDS = 5


def fit_raw_probe(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    c_value: float,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Fit on standardized training data and fold scaling into raw-space weights."""
    x_train = np.asarray(x[train], dtype=np.float32)
    mean = x_train.mean(axis=0, dtype=np.float64)
    scale = x_train.std(axis=0, dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    standardized = ((x_train - mean) / scale).astype(np.float32)
    model = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        dual=True,
        max_iter=5000,
        penalty="l2",
        random_state=seed,
        solver="liblinear",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
        model.fit(standardized, y[train])
    standardized_weights = np.asarray(model.coef_[0], dtype=np.float64)
    raw_weights = (standardized_weights / scale).astype(np.float32)
    raw_bias = float(model.intercept_[0] - np.dot(standardized_weights, mean / scale))
    if not np.isfinite(raw_weights).all() or not np.isfinite(raw_bias):
        raise RuntimeError("non-finite exported probe parameters")
    return raw_weights, raw_bias


def probe_votes(x: np.ndarray, probe: dict) -> np.ndarray:
    weights = np.asarray(probe["weights"], dtype=np.float32)
    bias = np.asarray(probe["bias"], dtype=np.float32)
    threshold = np.asarray(probe["threshold"], dtype=np.float32)
    margins = x @ weights.T + bias - threshold
    return (margins >= 0.0).reshape(-1)


def export(args: argparse.Namespace) -> dict:
    validation = json.loads(args.validation_result.read_text(encoding="utf-8"))
    if not validation.get("gate", {}).get("passed"):
        raise RuntimeError("E2 validation gate did not pass")
    if validation.get("experiment") != EXPECTED_EXPERIMENT:
        raise RuntimeError("unexpected validation experiment")
    if validation.get("layers") != EXPECTED_LAYERS:
        raise RuntimeError("validation layers differ from the frozen E2 layers")
    if float(validation.get("c_value")) != EXPECTED_C:
        raise RuntimeError("validation C differs from the frozen E2 value")
    if validation.get("seeds") != EXPECTED_SEEDS or validation.get("folds") != EXPECTED_FOLDS:
        raise RuntimeError("validation folds/seeds differ from the frozen E2 protocol")

    rows = base.read_metadata(args.internals / "metadata.csv")
    y = base.boolean_labels(rows)
    groups = np.asarray([row["problem_id"] for row in rows])
    layers = base.load_layers(args.internals, EXPECTED_LAYERS)
    expected_records = {
        (record["seed"], record["fold_index"]): record
        for record in validation["outer_folds"]
    }
    artifact_groups = []
    oof_votes: list[list[bool]] = [[] for _ in rows]
    every_index = np.arange(len(rows))

    for seed in EXPECTED_SEEDS:
        for fold_index, (train, test) in enumerate(
            base.split_indices(every_index, y, groups, EXPECTED_FOLDS, seed)
        ):
            probes = {}
            group_votes = []
            layer_scores = {}
            for layer_index in EXPECTED_LAYERS:
                weights, bias = fit_raw_probe(
                    layers[layer_index], y, train, EXPECTED_C, seed
                )
                probe = {
                    "weights": [weights.tolist()],
                    "bias": [bias],
                    "threshold": [0.0],
                }
                predictions = probe_votes(layers[layer_index][test], probe)
                group_votes.append(predictions)
                layer_scores[str(layer_index)] = float(
                    balanced_accuracy_score(y[test], predictions)
                )
                probes[str(layer_index)] = probe
            majority = np.mean(np.stack(group_votes), axis=0) >= 0.5
            group_score = float(balanced_accuracy_score(y[test], majority))
            expected = expected_records[(seed, fold_index)]["balanced_accuracy"]
            if abs(group_score - expected) > 1e-12:
                raise RuntimeError(
                    f"artifact replay mismatch seed={seed} fold={fold_index}: "
                    f"{group_score} != {expected}"
                )
            for index, layer_predictions in zip(
                test, np.stack(group_votes, axis=1), strict=True
            ):
                oof_votes[int(index)].extend(bool(value) for value in layer_predictions)
            train_ids = "\n".join(sorted(groups[train].tolist())).encode("utf-8")
            artifact_groups.append(
                {
                    "control_task": "NONE",
                    "seed": seed,
                    "fold_index": fold_index,
                    "train_count": len(train),
                    "test_count": len(test),
                    "train_problem_ids_sha256": hashlib.sha256(train_ids).hexdigest().upper(),
                    "validation_vote_balanced_accuracy": group_score,
                    "layer_validation_balanced_accuracy": layer_scores,
                    "probes": probes,
                }
            )
            print(
                f"exported seed={seed} fold={fold_index} vote_ba={group_score:.4f}",
                flush=True,
            )

    if any(len(votes) != len(EXPECTED_LAYERS) * len(EXPECTED_SEEDS) for votes in oof_votes):
        raise RuntimeError("exported OOF replay has incomplete vote coverage")
    oof_predictions = np.asarray([np.mean(votes) >= 0.5 for votes in oof_votes])
    replay_score = float(balanced_accuracy_score(y, oof_predictions))
    expected_score = float(validation["metrics"]["grouped_balanced_accuracy"])
    if abs(replay_score - expected_score) > 1e-12:
        raise RuntimeError(f"aggregate artifact replay mismatch: {replay_score} != {expected_score}")

    metrics = validation["metrics"]
    system_prompt = args.system_prompt.read_text(encoding="utf-8").strip()
    artifact = {
        "schema_version": 3,
        "artifact_type": "fixed_multilayer_vote_probe_ensemble",
        "model_id": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "system_prompt": system_prompt,
        "recommended_strategy": {
            "name": "fixed_multilayer_majority_vote",
            "layers": EXPECTED_LAYERS,
            "base_margin_threshold": 0.0,
            "positive_vote_fraction_threshold": 0.5,
            "tie_policy": "negative",
        },
        "training": {
            "selection_scope": "fixed_multilayer_problem_grouped",
            "c_value": EXPECTED_C,
            "folds": EXPECTED_FOLDS,
            "seeds": EXPECTED_SEEDS,
            "unique_problem_count": len(rows),
            "weight_dtype": "float32",
            "normalization": "per-layer training-fold standardization folded into weights",
        },
        "groups": artifact_groups,
        "validation_gate": {
            "grouped_balanced_accuracy": metrics["grouped_balanced_accuracy"],
            "bootstrap_95_lower": metrics["bootstrap_95_lower"],
            "selection_scope": "fixed_multilayer_problem_grouped",
            "source_holdout_balanced_accuracy": metrics[
                "source_holdout_balanced_accuracy"
            ],
            "randomized_control_balanced_accuracy": metrics[
                "randomized_control_balanced_accuracy"
            ],
            "randomized_control_advantage": metrics[
                "randomized_control_advantage"
            ],
            "interpretation": "sequential_exploratory",
        },
        "validation_replay": {
            "balanced_accuracy": replay_score,
            "predicted_positive": int(oof_predictions.sum()),
            "vote_count_per_problem": len(EXPECTED_LAYERS) * len(EXPECTED_SEEDS),
        },
        "provenance": {
            "source_internals_manifest": json.loads(
                (args.internals / "manifest.json").read_text(encoding="utf-8")
            ),
            "validation_result_sha256": hashlib.sha256(
                args.validation_result.read_bytes()
            ).hexdigest().upper(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(artifact, handle, protocol=4)
    artifact_hash = hashlib.sha256(args.output.read_bytes()).hexdigest().upper()
    summary = {
        "artifact": str(args.output.resolve()),
        "sha256": artifact_hash,
        "bytes": args.output.stat().st_size,
        "groups": len(artifact_groups),
        "layers": EXPECTED_LAYERS,
        "total_votes": len(artifact_groups) * len(EXPECTED_LAYERS),
        "oof_replay_balanced_accuracy": replay_score,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internals", type=Path, required=True)
    parser.add_argument("--validation-result", type=Path, required=True)
    parser.add_argument("--system-prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
