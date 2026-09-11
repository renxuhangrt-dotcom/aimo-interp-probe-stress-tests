#!/usr/bin/env python3
"""E2: fixed, equal-weight multi-layer probe vote with grouped evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

import train_nested_probe as base


def layer_vote(
    layers: dict[int, np.ndarray],
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    c_value: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    votes = []
    layer_scores = {}
    for layer_index in sorted(layers):
        margins = base.fit_margins(layers[layer_index], y, train, test, c_value, seed)
        predictions = margins >= 0.0
        votes.append(predictions.astype(np.float64))
        layer_scores[str(layer_index)] = float(balanced_accuracy_score(y[test], predictions))
    return np.mean(np.stack(votes), axis=0), layer_scores


def repeated_oof(
    layers: dict[int, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    folds: int,
    seeds: list[int],
    c_value: float,
) -> tuple[np.ndarray, list[dict]]:
    fractions: list[list[float]] = [[] for _ in range(len(y))]
    records = []
    indices = np.arange(len(y))
    for seed in seeds:
        for fold_index, (train, test) in enumerate(
            base.split_indices(indices, y, groups, folds, seed)
        ):
            vote_fraction, layer_scores = layer_vote(
                layers, y, train, test, c_value, seed
            )
            predictions = vote_fraction >= 0.5
            for index, fraction in zip(test, vote_fraction, strict=True):
                fractions[int(index)].append(float(fraction))
            record = {
                "seed": seed,
                "fold_index": fold_index,
                "train_count": len(train),
                "test_count": len(test),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y[test], predictions)
                ),
                "predicted_positive": int(predictions.sum()),
                "layer_balanced_accuracy": layer_scores,
            }
            records.append(record)
            print(
                f"seed={seed} fold={fold_index} "
                f"vote_ba={record['balanced_accuracy']:.4f} "
                f"positive={record['predicted_positive']}/{len(test)}",
                flush=True,
            )
    missing = [index for index, values in enumerate(fractions) if len(values) != len(seeds)]
    if missing:
        raise RuntimeError(f"OOF coverage failure: {missing[:10]}")
    return np.asarray([np.mean(values) for values in fractions]), records


def source_holdouts(
    layers: dict[int, np.ndarray],
    y: np.ndarray,
    sources: np.ndarray,
    c_value: float,
    seed: int,
) -> tuple[dict[str, float], list[dict]]:
    unique_sources = sorted(set(sources.tolist()))
    if len(unique_sources) != 2:
        raise ValueError(f"exactly two sources required, found {unique_sources}")
    scores = {}
    records = []
    for train_source, test_source in (
        (unique_sources[0], unique_sources[1]),
        (unique_sources[1], unique_sources[0]),
    ):
        train = np.flatnonzero(sources == train_source)
        test = np.flatnonzero(sources == test_source)
        fractions, layer_scores = layer_vote(layers, y, train, test, c_value, seed)
        predictions = fractions >= 0.5
        score = float(balanced_accuracy_score(y[test], predictions))
        direction = f"{train_source} -> {test_source}"
        scores[direction] = score
        records.append(
            {
                "direction": direction,
                "train_count": len(train),
                "test_count": len(test),
                "balanced_accuracy": score,
                "predicted_positive": int(predictions.sum()),
                "layer_balanced_accuracy": layer_scores,
            }
        )
        print(f"source holdout {direction}: vote_ba={score:.4f}", flush=True)
    return scores, records


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, str]],
    y: np.ndarray,
    vote_fractions: np.ndarray,
    result: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = vote_fractions >= 0.5
    with (output_dir / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "problem_id",
            "dataset_id",
            "label",
            "positive_vote_fraction",
            "prediction",
            "correct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row, label, fraction, prediction in zip(
            rows, y, vote_fractions, predictions, strict=True
        ):
            writer.writerow(
                {
                    "problem_id": row["problem_id"],
                    "dataset_id": row["dataset_id"],
                    "label": bool(label),
                    "positive_vote_fraction": f"{fraction:.12g}",
                    "prediction": bool(prediction),
                    "correct": bool(prediction == label),
                }
            )
    (output_dir / "validation_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    status = "PASS" if result["gate"]["passed"] else "FAIL"
    failures = "\n".join(f"- {item}" for item in result["gate"]["failures"]) or "- None"
    metrics = result["metrics"]
    report = f"""# E2 fixed multi-layer vote

Status: **{status}** (sequential exploratory validation)

- Layers: {result['layers']}
- Fixed C: {result['c_value']}
- Grouped OOF balanced accuracy: {metrics['grouped_balanced_accuracy']:.6f}
- Ordinary accuracy: {metrics['ordinary_accuracy']:.6f}
- Bootstrap 95% lower bound: {metrics['bootstrap_95_lower']:.6f}
- Randomized-label control: {metrics['randomized_control_balanced_accuracy']:.6f}
- Advantage over randomized control: {metrics['randomized_control_advantage']:.6f}
- Source holdouts: `{json.dumps(metrics['source_holdout_balanced_accuracy'], sort_keys=True)}`
- Predicted positives: {metrics['predicted_positive']} / {len(rows)}

## Gate failures

{failures}

The source-transfer data were observed during E1, so even a PASS here is not an
untouched confirmation. No artifact is exported by this script.
"""
    (output_dir / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internals", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--c-value", type=float, default=0.001)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260910)
    parser.add_argument("--min-score", type=float, default=0.65)
    parser.add_argument("--min-lower", type=float, default=0.55)
    parser.add_argument("--min-holdout", type=float, default=0.60)
    parser.add_argument("--min-control-advantage", type=float, default=0.05)
    args = parser.parse_args()

    rows = base.read_metadata(args.internals / "metadata.csv")
    y = base.boolean_labels(rows)
    groups = np.asarray([row["problem_id"] for row in rows])
    sources = np.asarray([row["dataset_id"] for row in rows])
    layers = base.load_layers(args.internals, args.layers)
    if len(args.layers) % 2 != 1:
        raise ValueError("an odd number of layers is required to avoid within-fold ties")

    fractions, fold_records = repeated_oof(
        layers, y, groups, args.folds, args.seeds, args.c_value
    )
    predictions = fractions >= 0.5
    grouped_score = float(balanced_accuracy_score(y, predictions))
    ordinary_accuracy = float(np.mean(y == predictions))
    lower = base.bootstrap_lower(
        y, predictions, args.bootstrap_resamples, args.bootstrap_seed
    )
    holdout_scores, holdout_records = source_holdouts(
        layers, y, sources, args.c_value, args.bootstrap_seed
    )

    random_y = np.random.default_rng(args.bootstrap_seed).permutation(y)
    random_fractions, random_records = repeated_oof(
        layers, random_y, groups, args.folds, [args.seeds[0]], args.c_value
    )
    random_score = float(
        balanced_accuracy_score(random_y, random_fractions >= 0.5)
    )
    advantage = grouped_score - random_score

    failures = []
    if grouped_score < args.min_score:
        failures.append(f"grouped balanced accuracy {grouped_score:.6f} < {args.min_score:.2f}")
    if lower <= args.min_lower:
        failures.append(f"bootstrap lower bound {lower:.6f} is not > {args.min_lower:.2f}")
    for direction, score in holdout_scores.items():
        if score < args.min_holdout:
            failures.append(f"source holdout {direction} {score:.6f} < {args.min_holdout:.2f}")
    if advantage < args.min_control_advantage:
        failures.append(
            f"random-control advantage {advantage:.6f} < {args.min_control_advantage:.2f}"
        )

    tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
    result = {
        "schema_version": 1,
        "experiment": "E2_fixed_multilayer_vote",
        "selection_scope": "fixed_multilayer_problem_grouped",
        "interpretation": "sequential_exploratory",
        "layers": sorted(layers),
        "c_value": args.c_value,
        "folds": args.folds,
        "seeds": args.seeds,
        "metrics": {
            "grouped_balanced_accuracy": grouped_score,
            "ordinary_accuracy": ordinary_accuracy,
            "bootstrap_95_lower": lower,
            "source_holdout_balanced_accuracy": holdout_scores,
            "randomized_control_balanced_accuracy": random_score,
            "randomized_control_advantage": advantage,
            "predicted_positive": int(predictions.sum()),
            "actual_positive": int(y.sum()),
            "confusion_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
        },
        "outer_folds": fold_records,
        "source_holdouts": holdout_records,
        "randomized_control_folds": random_records,
        "gate": {
            "passed": not failures,
            "failures": failures,
            "thresholds": {
                "grouped_balanced_accuracy_min": args.min_score,
                "bootstrap_95_lower_strictly_above": args.min_lower,
                "source_holdout_balanced_accuracy_each_min": args.min_holdout,
                "randomized_control_advantage_min": args.min_control_advantage,
            },
        },
    }
    write_outputs(args.output_dir, rows, y, fractions, result)
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))
    print(f"GATE: {'PASS' if not failures else 'FAIL'}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
