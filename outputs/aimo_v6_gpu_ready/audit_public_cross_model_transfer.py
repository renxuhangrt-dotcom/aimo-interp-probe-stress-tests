#!/usr/bin/env python3
"""Audit frozen V6 predictions against the public multi-model sample."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
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


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def accuracy(labels: list[bool], predictions: list[bool]) -> float:
    return float(np.mean(np.asarray(labels) == np.asarray(predictions)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--internals", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--training-module", type=Path, required=True)
    parser.add_argument("--public-input", type=Path, required=True)
    parser.add_argument("--public-labels", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    make_inference_importable()
    inference = load_module("v6_cross_model_inference", args.inference)
    training = load_module("v6_cross_model_training", args.training_module)
    artifact = inference.load_pickle_artifact(args.artifact)
    layers = inference._required_layer_indices(artifact)
    training_rows = training.read_metadata(args.internals / "metadata.csv")
    matrices = training.load_layers(args.internals, layers)
    problem_to_index = {row["original_problem"]: index for index, row in enumerate(training_rows)}

    cases = read_jsonl(args.public_input)
    labels_by_id = {row["id"]: bool(row["is_robust"]) for row in read_jsonl(args.public_labels)}
    missing = [row["id"] for row in cases if row["problem"] not in problem_to_index]
    if missing:
        raise RuntimeError(f"public problems missing from frozen internals: {missing}")

    records = []
    for case in cases:
        row_index = problem_to_index[case["problem"]]
        vectors = {
            layer: np.asarray(matrices[layer][row_index], dtype=np.float32)
            for layer in layers
        }
        fraction = inference.multilayer_positive_vote_fraction(vectors, artifact)
        records.append(
            {
                "id": case["id"],
                "problem_key": hashlib.sha256(case["problem"].encode("utf-8")).hexdigest()[:12],
                "model_id": case["model_id"],
                "label": labels_by_id[case["id"]],
                "prediction": bool(fraction >= 0.5),
                "positive_vote_fraction": fraction,
            }
        )

    labels = [row["label"] for row in records]
    predictions = [row["prediction"] for row in records]
    per_model_rows: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        per_model_rows[row["model_id"]].append(row)
    per_model = {}
    for model_id, rows in sorted(per_model_rows.items()):
        model_labels = [row["label"] for row in rows]
        model_predictions = [row["prediction"] for row in rows]
        per_model[model_id] = {
            "cases": len(rows),
            "positive_labels": int(sum(model_labels)),
            "positive_predictions": int(sum(model_predictions)),
            "accuracy": accuracy(model_labels, model_predictions),
        }

    non_training_model_records = [row for row in records if row["model_id"] != "qwen3-8b:low"]
    transfer_labels = [row["label"] for row in non_training_model_records]
    transfer_predictions = [row["prediction"] for row in non_training_model_records]
    fractions = np.asarray([row["positive_vote_fraction"] for row in records])
    per_problem_rows: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        per_problem_rows[row["problem_key"]].append(row)
    per_problem = {
        key: {
            "cases": len(rows),
            "positive_labels": int(sum(row["label"] for row in rows)),
            "positive_vote_fraction": float(rows[0]["positive_vote_fraction"]),
            "prediction": bool(rows[0]["prediction"]),
        }
        for key, rows in sorted(per_problem_rows.items())
    }
    result = {
        "schema_version": 1,
        "method": "frozen_v6_problem_proxy_cross_model_audit",
        "scientific_role": "diagnostic_only_no_tuning",
        "cases": len(records),
        "unique_problems": len({case["problem"] for case in cases}),
        "models": len(per_model),
        "all_cases": {
            "accuracy": accuracy(labels, predictions),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "always_false_accuracy": accuracy(labels, [False] * len(labels)),
            "positive_labels": int(sum(labels)),
            "positive_predictions": int(sum(predictions)),
            "positive_vote_fraction_min": float(fractions.min()),
            "positive_vote_fraction_median": float(np.median(fractions)),
            "positive_vote_fraction_mean": float(fractions.mean()),
            "positive_vote_fraction_max": float(fractions.max()),
        },
        "non_qwen_training_model_cases": {
            "cases": len(non_training_model_records),
            "accuracy": accuracy(transfer_labels, transfer_predictions),
            "balanced_accuracy": float(
                balanced_accuracy_score(transfer_labels, transfer_predictions)
            ),
            "always_false_accuracy": accuracy(
                transfer_labels, [False] * len(transfer_labels)
            ),
        },
        "per_model": per_model,
        "per_problem": per_problem,
        "limitations": [
            "The public problem texts are evaluated with frozen DeepSeek representations; labels for the other model IDs were not used for training.",
            "The public sample is tiny and its model IDs do not exactly match every currently cached evaluation checkpoint.",
            "This audit freezes the V6 artifact and threshold; it must not be used for per-case or leaderboard-driven tuning.",
        ],
    }
    args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Frozen V6 public cross-model transfer audit",
        "",
        "Decision: **NO-GO for a direct Main Track proxy submission**.",
        "",
        "This is a diagnostic-only audit. No V6 weights, layers, votes, or thresholds were changed.",
        "",
        "| Slice | Cases | Accuracy | Balanced accuracy | Always-false accuracy |",
        "|---|---:|---:|---:|---:|",
        f"| All public multi-model cases | {len(records)} | {result['all_cases']['accuracy']:.4f} | {result['all_cases']['balanced_accuracy']:.4f} | {result['all_cases']['always_false_accuracy']:.4f} |",
        f"| Excluding qwen3-8b training-model case | {len(non_training_model_records)} | {result['non_qwen_training_model_cases']['accuracy']:.4f} | {result['non_qwen_training_model_cases']['balanced_accuracy']:.4f} | {result['non_qwen_training_model_cases']['always_false_accuracy']:.4f} |",
        "",
        "All 24 cases were predicted positive. The positive-vote fractions span "
        f"`{fractions.min():.4f}` to `{fractions.max():.4f}` with median "
        f"`{np.median(fractions):.4f}`.",
        "",
        "## Per model",
        "",
        "| Model ID | Cases | Positive labels | Positive predictions | Accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_id, metrics in per_model.items():
        lines.append(
            f"| `{model_id}` | {metrics['cases']} | {metrics['positive_labels']} | "
            f"{metrics['positive_predictions']} | {metrics['accuracy']:.4f} |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in result["limitations"])
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
