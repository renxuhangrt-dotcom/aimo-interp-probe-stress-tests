#!/usr/bin/env python3
"""E8: fixed equal vote over final problem-token representations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

import train_layer_vote_probe as vote
import train_nested_probe as base


EXPECTED_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32, 36]
EXPECTED_MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
EXPECTED_DATASET_HASH = "3FC14BFC02F7C1D0D79BB917491C7710F6A798150416B157C6C34205CEC9E067"


def audit_internals(internals: Path, rows: list[dict[str, str]]) -> dict:
    manifest_path = internals / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("problem-last internals are missing manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "representation": "problem_last",
        "model_id": EXPECTED_MODEL_ID,
        "problem_count": 137,
        "layer_count": 37,
        "hidden_dimension": 4096,
        "prepared_dataset_canonical_sha256": EXPECTED_DATASET_HASH,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    views = {row.get("extraction_view") for row in rows}
    if views != {"problem_last"}:
        mismatches["metadata.extraction_view"] = {
            "expected": ["problem_last"],
            "actual": sorted(str(value) for value in views),
        }
    if mismatches:
        raise RuntimeError(f"problem-last internals audit failed: {mismatches}")
    return manifest


def read_baseline_predictions(path: Path, rows: list[dict[str, str]]) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        values = list(csv.DictReader(handle))
    by_id = {row["problem_id"]: row for row in values}
    ids = [row["problem_id"] for row in rows]
    if len(by_id) != len(ids) or set(by_id) != set(ids):
        raise ValueError("baseline OOF problem IDs do not match metadata")
    return np.asarray(
        [by_id[problem_id]["prediction"].strip().lower() == "true" for problem_id in ids],
        dtype=bool,
    )


def paired_bootstrap_difference_lower(
    y: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    count: int,
    seed: int,
) -> tuple[float, float, float]:
    generator = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(count):
        sample = generator.integers(0, len(y), size=len(y))
        if len(np.unique(y[sample])) < 2:
            continue
        differences.append(
            float(
                balanced_accuracy_score(y[sample], candidate[sample])
                - balanced_accuracy_score(y[sample], baseline[sample])
            )
        )
    if not differences:
        raise RuntimeError("paired bootstrap produced no valid resamples")
    lower, median, upper = np.quantile(differences, [0.025, 0.5, 0.975])
    return float(lower), float(median), float(upper)


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, str]],
    y: np.ndarray,
    fractions: np.ndarray,
    baseline: np.ndarray,
    result: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = fractions >= 0.5
    with (output_dir / "oof_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "problem_id",
            "dataset_id",
            "label",
            "positive_problem_last_vote_fraction",
            "prediction",
            "v6_prediction",
            "changed_vs_v6",
            "correct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row, label, fraction, prediction, old_prediction in zip(
            rows, y, fractions, predictions, baseline, strict=True
        ):
            writer.writerow(
                {
                    "problem_id": row["problem_id"],
                    "dataset_id": row["dataset_id"],
                    "label": bool(label),
                    "positive_problem_last_vote_fraction": f"{fraction:.12g}",
                    "prediction": bool(prediction),
                    "v6_prediction": bool(old_prediction),
                    "changed_vs_v6": bool(prediction != old_prediction),
                    "correct": bool(prediction == label),
                }
            )
    (output_dir / "validation_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metrics = result["metrics"]
    status = "PASS" if result["gate"]["passed"] else "FAIL"
    failures = "\n".join(f"- {item}" for item in result["gate"]["failures"]) or "- None"
    report = f"""# E8 final problem-token representation vote

Status: **{status}** (sequential exploratory validation)

- Representation: `problem_last`
- Layers: {result['layers']}
- Fixed C: {result['c_value']}
- Grouped OOF balanced accuracy: {metrics['grouped_balanced_accuracy']:.6f}
- V6 grouped OOF balanced accuracy: {metrics['v6_grouped_balanced_accuracy']:.6f}
- Difference versus V6: {metrics['balanced_accuracy_difference_vs_v6']:.6f}
- Candidate bootstrap 95% lower: {metrics['bootstrap_95_lower']:.6f}
- Paired bootstrap difference 95% interval: {metrics['paired_bootstrap_difference_95']}
- Source holdouts: `{json.dumps(metrics['source_holdout_balanced_accuracy'], sort_keys=True)}`
- Mean source holdout: {metrics['mean_source_holdout_balanced_accuracy']:.6f}
- Randomized-label advantage: {metrics['randomized_control_advantage']:.6f}
- Disagreement versus V6: {metrics['disagreement_vs_v6']:.6f}
- Corrected/broken versus V6: {metrics['corrected_vs_v6']} / {metrics['broken_vs_v6']}
- Predicted positives: {metrics['predicted_positive']} / {len(rows)}

## Gate failures

{failures}

No artifact is exported by this validation script.
"""
    (output_dir / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internals", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-oof", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=EXPECTED_LAYERS)
    parser.add_argument("--c-value", type=float, default=0.001)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260910)
    parser.add_argument("--min-improvement", type=float, default=0.02)
    parser.add_argument("--v6-bootstrap-lower", type=float, default=0.6169437631511935)
    parser.add_argument("--min-holdout", type=float, default=0.65)
    parser.add_argument("--min-mean-holdout", type=float, default=0.69)
    parser.add_argument("--min-control-advantage", type=float, default=0.10)
    parser.add_argument("--min-disagreement", type=float, default=0.08)
    args = parser.parse_args()

    if args.layers != EXPECTED_LAYERS:
        raise ValueError("E8 layers differ from the preregistered sequence")
    rows = base.read_metadata(args.internals / "metadata.csv")
    internals_manifest = audit_internals(args.internals, rows)
    y = base.boolean_labels(rows)
    groups = np.asarray([row["problem_id"] for row in rows])
    sources = np.asarray([row["dataset_id"] for row in rows])
    layers = base.load_layers(args.internals, args.layers)
    baseline = read_baseline_predictions(args.baseline_oof, rows)

    fractions, fold_records = vote.repeated_oof(
        layers, y, groups, args.folds, args.seeds, args.c_value
    )
    predictions = fractions >= 0.5
    grouped_score = float(balanced_accuracy_score(y, predictions))
    baseline_score = float(balanced_accuracy_score(y, baseline))
    improvement = grouped_score - baseline_score
    ordinary_accuracy = float(np.mean(y == predictions))
    lower = base.bootstrap_lower(
        y, predictions, args.bootstrap_resamples, args.bootstrap_seed
    )
    paired_lower, paired_median, paired_upper = paired_bootstrap_difference_lower(
        y,
        predictions,
        baseline,
        args.bootstrap_resamples,
        args.bootstrap_seed,
    )
    holdout_scores, holdout_records = vote.source_holdouts(
        layers, y, sources, args.c_value, args.bootstrap_seed
    )
    mean_holdout = float(np.mean(list(holdout_scores.values())))

    random_y = np.random.default_rng(args.bootstrap_seed).permutation(y)
    random_fractions, random_records = vote.repeated_oof(
        layers, random_y, groups, args.folds, [args.seeds[0]], args.c_value
    )
    random_score = float(
        balanced_accuracy_score(random_y, random_fractions >= 0.5)
    )
    advantage = grouped_score - random_score
    disagreement = float(np.mean(predictions != baseline))
    corrected = int(np.sum((predictions == y) & (baseline != y)))
    broken = int(np.sum((predictions != y) & (baseline == y)))

    failures: list[str] = []
    if improvement < args.min_improvement:
        failures.append(
            f"balanced-accuracy improvement {improvement:.6f} < {args.min_improvement:.2f}"
        )
    if lower <= args.v6_bootstrap_lower:
        failures.append(
            f"bootstrap lower {lower:.6f} is not > V6 {args.v6_bootstrap_lower:.6f}"
        )
    if paired_lower <= 0.0:
        failures.append(
            f"paired bootstrap difference lower {paired_lower:.6f} is not > 0"
        )
    for direction, score in holdout_scores.items():
        if score < args.min_holdout:
            failures.append(
                f"source holdout {direction} {score:.6f} < {args.min_holdout:.2f}"
            )
    if mean_holdout < args.min_mean_holdout:
        failures.append(
            f"mean source holdout {mean_holdout:.6f} < {args.min_mean_holdout:.2f}"
        )
    if advantage < args.min_control_advantage:
        failures.append(
            f"random-control advantage {advantage:.6f} < {args.min_control_advantage:.2f}"
        )
    if disagreement < args.min_disagreement:
        failures.append(
            f"OOF disagreement versus V6 {disagreement:.6f} < {args.min_disagreement:.2f}"
        )

    tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
    result = {
        "schema_version": 1,
        "experiment": "E8_problem_last_token_vote",
        "selection_scope": "fixed_problem_last_layer_vote_grouped",
        "interpretation": "sequential_exploratory",
        "representation": "problem_last",
        "layers": args.layers,
        "internals_manifest": internals_manifest,
        "c_value": args.c_value,
        "folds": args.folds,
        "seeds": args.seeds,
        "metrics": {
            "grouped_balanced_accuracy": grouped_score,
            "v6_grouped_balanced_accuracy": baseline_score,
            "balanced_accuracy_difference_vs_v6": improvement,
            "ordinary_accuracy": ordinary_accuracy,
            "bootstrap_95_lower": lower,
            "paired_bootstrap_difference_95": [paired_lower, paired_median, paired_upper],
            "source_holdout_balanced_accuracy": holdout_scores,
            "mean_source_holdout_balanced_accuracy": mean_holdout,
            "randomized_control_balanced_accuracy": random_score,
            "randomized_control_advantage": advantage,
            "disagreement_vs_v6": disagreement,
            "corrected_vs_v6": corrected,
            "broken_vs_v6": broken,
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
                "balanced_accuracy_improvement_vs_v6_min": args.min_improvement,
                "candidate_bootstrap_lower_strictly_above_v6": args.v6_bootstrap_lower,
                "paired_bootstrap_difference_lower_strictly_above": 0.0,
                "source_holdout_each_min": args.min_holdout,
                "mean_source_holdout_min": args.min_mean_holdout,
                "randomized_control_advantage_min": args.min_control_advantage,
                "disagreement_vs_v6_min": args.min_disagreement,
            },
        },
    }
    write_outputs(args.output_dir, rows, y, fractions, baseline, result)
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))
    print(f"GATE: {'PASS' if not failures else 'FAIL'}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
