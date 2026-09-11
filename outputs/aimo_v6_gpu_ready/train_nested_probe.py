#!/usr/bin/env python3
"""Nested problem-grouped validation for the AIMO robustness probe.

This script is deliberately validation-only.  It writes OOF predictions and a
fail-closed gate decision, but it does not create a submission artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("metadata is empty")
    ids = [row["problem_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("metadata must contain exactly one row per problem_id")
    return rows


def boolean_labels(rows: list[dict[str, str]]) -> np.ndarray:
    values = []
    for row in rows:
        normalized = row["model_is_robust"].strip().lower()
        if normalized not in {"true", "false"}:
            raise ValueError(f"invalid label {row['model_is_robust']!r}")
        values.append(normalized == "true")
    return np.asarray(values, dtype=np.int8)


def safe_folds(y: np.ndarray, requested: int) -> int:
    counts = np.bincount(y.astype(int), minlength=2)
    result = min(requested, int(counts.min()))
    if result < 2:
        raise ValueError(f"not enough examples per class for grouped CV: {counts.tolist()}")
    return result


def split_indices(
    indices: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    folds: int,
    seed: int,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    local_y = y[indices]
    local_groups = groups[indices]
    splitter = StratifiedGroupKFold(
        n_splits=safe_folds(local_y, folds), shuffle=True, random_state=seed
    )
    dummy = np.zeros((len(indices), 1), dtype=np.float32)
    for local_train, local_test in splitter.split(dummy, local_y, local_groups):
        train = indices[local_train]
        test = indices[local_test]
        overlap = set(groups[train]).intersection(groups[test])
        if overlap:
            raise RuntimeError(f"group leakage detected: {sorted(overlap)[:5]}")
        yield train, test


def normalize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, dtype=np.float64)
    scale = x_train.std(axis=0, dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    return (
        ((x_train - mean) / scale).astype(np.float32),
        ((x_test - mean) / scale).astype(np.float32),
    )


def fit_margins(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    c_value: float,
    seed: int,
) -> np.ndarray:
    x_train, x_test = normalize_train_test(x[train], x[test])
    model = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        dual=True,
        max_iter=5000,
        penalty="l2",
        random_state=seed,
        solver="liblinear",
    )
    # scikit-learn 1.8 deprecates the explicit penalty spelling used by older
    # supported versions.  Keep the cross-version arguments and avoid flooding
    # long GPU-run logs with one warning per fit.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
        model.fit(x_train, y[train])
    return np.asarray(model.decision_function(x_test), dtype=np.float64)


def select_candidate(
    layers: dict[int, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    train_pool: np.ndarray,
    candidate_layers: list[int],
    c_values: list[float],
    inner_folds: int,
    seed: int,
) -> tuple[int, float, float]:
    splits = list(split_indices(train_pool, y, groups, inner_folds, seed + 100_000))
    candidates: list[tuple[float, float, int]] = []
    for layer_index in candidate_layers:
        x = layers[layer_index]
        for c_value in c_values:
            fold_scores = []
            for inner_train, inner_test in splits:
                margins = fit_margins(x, y, inner_train, inner_test, c_value, seed)
                fold_scores.append(
                    balanced_accuracy_score(y[inner_test], margins >= 0.0)
                )
            candidates.append((float(np.mean(fold_scores)), c_value, layer_index))
    # Fixed tie-break: best score, then stronger regularization, then earlier layer.
    score, c_value, layer_index = min(candidates, key=lambda item: (-item[0], item[1], item[2]))
    return layer_index, c_value, score


def nested_oof(
    layers: dict[int, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    candidate_layers: list[int],
    c_values: list[float],
    outer_folds: int,
    inner_folds: int,
    seeds: list[int],
) -> tuple[np.ndarray, list[dict]]:
    all_margins: list[list[float]] = [[] for _ in range(len(y))]
    records: list[dict] = []
    every_index = np.arange(len(y))
    for seed in seeds:
        for fold_index, (train, test) in enumerate(
            split_indices(every_index, y, groups, outer_folds, seed)
        ):
            layer_index, c_value, inner_score = select_candidate(
                layers,
                y,
                groups,
                train,
                candidate_layers,
                c_values,
                inner_folds,
                seed + fold_index,
            )
            margins = fit_margins(layers[layer_index], y, train, test, c_value, seed)
            for index, margin in zip(test, margins, strict=True):
                all_margins[int(index)].append(float(margin))
            records.append(
                {
                    "seed": seed,
                    "fold_index": fold_index,
                    "train_count": len(train),
                    "test_count": len(test),
                    "selected_layer": layer_index,
                    "selected_c": c_value,
                    "inner_balanced_accuracy": inner_score,
                    "outer_balanced_accuracy": float(
                        balanced_accuracy_score(y[test], margins >= 0.0)
                    ),
                }
            )
            print(
                f"seed={seed} fold={fold_index} layer={layer_index} C={c_value:g} "
                f"inner={inner_score:.4f} outer={records[-1]['outer_balanced_accuracy']:.4f}",
                flush=True,
            )
    missing = [index for index, margins in enumerate(all_margins) if len(margins) != len(seeds)]
    if missing:
        raise RuntimeError(f"OOF coverage failure at indices: {missing[:10]}")
    return np.asarray([np.mean(values) for values in all_margins]), records


def source_holdouts(
    layers: dict[int, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    sources: np.ndarray,
    candidate_layers: list[int],
    c_values: list[float],
    inner_folds: int,
    seed: int,
) -> tuple[dict[str, float], list[dict]]:
    unique_sources = sorted(set(sources.tolist()))
    if len(unique_sources) != 2:
        raise ValueError(f"exactly two sources are required, found {unique_sources}")
    scores: dict[str, float] = {}
    records = []
    for train_source, test_source in (
        (unique_sources[0], unique_sources[1]),
        (unique_sources[1], unique_sources[0]),
    ):
        train = np.flatnonzero(sources == train_source)
        test = np.flatnonzero(sources == test_source)
        layer_index, c_value, inner_score = select_candidate(
            layers,
            y,
            groups,
            train,
            candidate_layers,
            c_values,
            inner_folds,
            seed,
        )
        margins = fit_margins(layers[layer_index], y, train, test, c_value, seed)
        score = float(balanced_accuracy_score(y[test], margins >= 0.0))
        direction = f"{train_source} -> {test_source}"
        scores[direction] = score
        records.append(
            {
                "direction": direction,
                "train_count": len(train),
                "test_count": len(test),
                "selected_layer": layer_index,
                "selected_c": c_value,
                "inner_balanced_accuracy": inner_score,
                "holdout_balanced_accuracy": score,
            }
        )
        print(f"source holdout {direction}: {score:.4f}", flush=True)
    return scores, records


def bootstrap_lower(y: np.ndarray, predictions: np.ndarray, count: int, seed: int) -> float:
    generator = np.random.default_rng(seed)
    scores = []
    for _ in range(count):
        sample = generator.integers(0, len(y), size=len(y))
        if len(np.unique(y[sample])) < 2:
            continue
        scores.append(balanced_accuracy_score(y[sample], predictions[sample]))
    if not scores:
        raise RuntimeError("bootstrap produced no valid two-class resamples")
    return float(np.quantile(scores, 0.025))


def load_layers(internals: Path, requested: list[int] | None) -> dict[int, np.ndarray]:
    available = {
        int(path.stem.split("_")[-1]): path for path in sorted(internals.glob("layer_*.npy"))
    }
    if not available:
        raise FileNotFoundError(f"no layer_*.npy files found in {internals}")
    selected = sorted(available) if requested is None else requested
    missing = set(selected).difference(available)
    if missing:
        raise ValueError(f"requested layers not found: {sorted(missing)}")
    result = {index: np.load(available[index], mmap_mode="r") for index in selected}
    row_counts = {matrix.shape[0] for matrix in result.values()}
    if len(row_counts) != 1:
        raise ValueError(f"layer row counts disagree: {sorted(row_counts)}")
    return result


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, str]],
    y: np.ndarray,
    margins: np.ndarray,
    result: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = margins >= 0.0
    with (output_dir / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "problem_id",
                "dataset_id",
                "label",
                "mean_oof_margin",
                "prediction",
                "correct",
            ],
        )
        writer.writeheader()
        for row, label, margin, prediction in zip(rows, y, margins, predictions, strict=True):
            writer.writerow(
                {
                    "problem_id": row["problem_id"],
                    "dataset_id": row["dataset_id"],
                    "label": bool(label),
                    "mean_oof_margin": f"{margin:.12g}",
                    "prediction": bool(prediction),
                    "correct": bool(prediction == label),
                }
            )
    (output_dir / "validation_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    status = "PASS" if result["gate"]["passed"] else "FAIL"
    failures = "\n".join(f"- {item}" for item in result["gate"]["failures"]) or "- None"
    report = f"""# Nested grouped probe validation

Status: **{status}**

- Unique problems: {len(rows)}
- Nested OOF balanced accuracy: {result['metrics']['grouped_balanced_accuracy']:.6f}
- Grouped bootstrap 95% lower bound: {result['metrics']['bootstrap_95_lower']:.6f}
- Randomized-label control: {result['metrics']['randomized_control_balanced_accuracy']:.6f}
- Advantage over randomized control: {result['metrics']['randomized_control_advantage']:.6f}
- Source holdouts: `{json.dumps(result['metrics']['source_holdout_balanced_accuracy'], sort_keys=True)}`

## Gate failures

{failures}

No V6 artifact may be exported unless this report is PASS.
"""
    (output_dir / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internals", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="*")
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260910)
    parser.add_argument("--min-score", type=float, default=0.65)
    parser.add_argument("--min-lower", type=float, default=0.55)
    parser.add_argument("--min-holdout", type=float, default=0.60)
    parser.add_argument("--min-control-advantage", type=float, default=0.05)
    args = parser.parse_args()

    rows = read_metadata(args.internals / "metadata.csv")
    y = boolean_labels(rows)
    groups = np.asarray([row["problem_id"] for row in rows])
    sources = np.asarray([row["dataset_id"] for row in rows])
    layers = load_layers(args.internals, args.layers)
    if next(iter(layers.values())).shape[0] != len(rows):
        raise ValueError("metadata and layer row counts disagree")
    candidate_layers = sorted(layers)
    print(f"loaded {len(rows)} problems and layers {candidate_layers}", flush=True)

    margins, fold_records = nested_oof(
        layers,
        y,
        groups,
        candidate_layers,
        args.c_values,
        args.outer_folds,
        args.inner_folds,
        args.seeds,
    )
    predictions = margins >= 0.0
    grouped_score = float(balanced_accuracy_score(y, predictions))
    lower = bootstrap_lower(y, predictions, args.bootstrap_resamples, args.bootstrap_seed)
    holdout_scores, holdout_records = source_holdouts(
        layers,
        y,
        groups,
        sources,
        candidate_layers,
        args.c_values,
        args.inner_folds,
        args.bootstrap_seed,
    )

    random_y = np.random.default_rng(args.bootstrap_seed).permutation(y)
    random_margins, random_records = nested_oof(
        layers,
        random_y,
        groups,
        candidate_layers,
        args.c_values,
        args.outer_folds,
        args.inner_folds,
        [args.seeds[0]],
    )
    random_score = float(balanced_accuracy_score(random_y, random_margins >= 0.0))
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

    result = {
        "schema_version": 1,
        "selection_scope": "nested_problem_grouped",
        "counts": {
            "problems": len(rows),
            "labels": dict(sorted(Counter(map(str, y.astype(bool))).items())),
            "sources": dict(sorted(Counter(sources.tolist()).items())),
        },
        "candidate_layers": candidate_layers,
        "c_values": args.c_values,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "seeds": args.seeds,
        "metrics": {
            "grouped_balanced_accuracy": grouped_score,
            "bootstrap_95_lower": lower,
            "source_holdout_balanced_accuracy": holdout_scores,
            "randomized_control_balanced_accuracy": random_score,
            "randomized_control_advantage": advantage,
        },
        "nested_folds": fold_records,
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
    write_outputs(args.output_dir, rows, y, margins, result)
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))
    print(f"GATE: {'PASS' if not failures else 'FAIL'}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
