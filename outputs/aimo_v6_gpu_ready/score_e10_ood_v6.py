#!/usr/bin/env python3
"""Score a returned E10 internals archive with the frozen V6 artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import pickle
import zipfile
from pathlib import Path

import numpy as np


V6_ARTIFACT_SHA256 = "7A4DF77D031530DC4998CBC4B6FFF2B8B717F99C7FA04E2646847C6A5BE3672B"
E10_DATASET_REVISION = "c0ffb7e678294cc8819d50d979ba28ba49bd02d0"
E10_DATASET_SHA256 = "2989AB6A018F2779A821209F414BD0049C9EF1C2488F589FF2D3BF4ED16942FF"
EXPECTED_MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
EXPECTED_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32, 36]
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 250_000_000
MAX_ENTRY_BYTES = 50_000_000


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def index_archive(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    entries = {}
    total = 0
    for info in archive.infolist():
        normalized = info.filename.replace("\\", "/").lstrip("./")
        if not normalized or normalized.endswith("/"):
            continue
        if normalized.startswith("/") or ".." in Path(normalized).parts:
            raise ValueError(f"unsafe archive member: {info.filename!r}")
        if info.file_size > MAX_ENTRY_BYTES:
            raise ValueError(f"archive member is unexpectedly large: {info.filename!r}")
        total += info.file_size
        if total > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise ValueError("archive is unexpectedly large")
        if normalized in entries:
            raise ValueError(f"duplicate archive member: {normalized!r}")
        entries[normalized] = info
    return entries


def find_entry(entries: dict[str, zipfile.ZipInfo], suffix: str) -> zipfile.ZipInfo:
    suffix = suffix.replace("\\", "/")
    matches = [info for name, info in entries.items() if name == suffix or name.endswith("/" + suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one archive entry ending in {suffix!r}, found {len(matches)}")
    return matches[0]


def read_json(archive: zipfile.ZipFile, entries: dict[str, zipfile.ZipInfo], suffix: str) -> dict:
    value = json.loads(archive.read(find_entry(entries, suffix)).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{suffix} must contain a JSON object")
    return value


def load_e10(path: Path) -> tuple[list[dict[str, str]], dict[int, np.ndarray], dict, dict]:
    with zipfile.ZipFile(path) as archive:
        entries = index_archive(archive)
        manifest = read_json(archive, entries, "manifest.json")
        audit = read_json(archive, entries, "dataset_audit.json")
        metadata_payload = archive.read(find_entry(entries, "metadata.csv")).decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(metadata_payload)))
        if not rows:
            raise ValueError("E10 metadata is empty")
        layers = {}
        for layer in EXPECTED_LAYERS:
            payload = archive.read(find_entry(entries, f"layer_{layer:03d}.npy"))
            matrix = np.load(io.BytesIO(payload), allow_pickle=False)
            matrix = np.asarray(matrix, dtype=np.float32)
            if matrix.shape != (len(rows), 4096) or not np.isfinite(matrix).all():
                raise ValueError(f"invalid layer {layer} matrix: {matrix.shape}")
            layers[layer] = matrix

    if manifest.get("notebook_version") != "e10-ood-small-1":
        raise ValueError(f"unexpected E10 notebook version: {manifest.get('notebook_version')!r}")
    if manifest.get("model_id") != EXPECTED_MODEL_ID:
        raise ValueError(f"unexpected extraction checkpoint: {manifest.get('model_id')!r}")
    if int(manifest.get("problem_count", -1)) != len(rows):
        raise ValueError("manifest problem count does not match metadata")
    if audit.get("source_revision") != E10_DATASET_REVISION:
        raise ValueError("E10 dataset revision mismatch")
    if audit.get("source_file_sha256") != E10_DATASET_SHA256:
        raise ValueError("E10 dataset hash mismatch")
    if audit.get("selected_model_id") != "qwen3-8b:low":
        raise ValueError("E10 selected-model mismatch")
    if audit.get("training_problem_id_overlap") or audit.get("training_normalized_text_overlap"):
        raise ValueError("E10 audit reports training/OOD overlap")
    rule = audit.get("label_rule_audit") or {}
    if rule.get("unique_problems") != 137 or rule.get("matching_labels") != 137:
        raise ValueError("E10 label rule did not reproduce all training labels")
    return rows, layers, manifest, audit


def load_v6_artifact(v6_zip: Path) -> dict:
    with zipfile.ZipFile(v6_zip) as archive:
        payload = archive.read("probe_artifacts/probe_artifact.pkl")
    actual = sha256_bytes(payload)
    if actual != V6_ARTIFACT_SHA256:
        raise ValueError(f"V6 artifact hash mismatch: {actual}")
    artifact = pickle.loads(payload)
    if not isinstance(artifact, dict):
        raise ValueError("V6 artifact is not a dictionary")
    strategy = artifact.get("recommended_strategy") or {}
    if artifact.get("schema_version") != 3 or strategy.get("layers") != EXPECTED_LAYERS:
        raise ValueError("unexpected V6 schema or layer set")
    groups = [group for group in artifact.get("groups", []) if group.get("control_task") == "NONE"]
    if len(groups) != 25:
        raise ValueError(f"expected 25 V6 inference groups, found {len(groups)}")
    return artifact


def predict(artifact: dict, layers: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    groups = [group for group in artifact["groups"] if group.get("control_task") == "NONE"]
    vote_columns = []
    for group in groups:
        probes = group.get("probes") or {}
        for layer in EXPECTED_LAYERS:
            probe = probes.get(layer, probes.get(str(layer)))
            if not isinstance(probe, dict):
                raise ValueError(f"artifact group is missing layer {layer}")
            weights = np.asarray(probe["weights"], dtype=np.float32)
            if weights.ndim == 1:
                weights = weights.reshape(1, -1)
            bias = np.asarray(probe["bias"], dtype=np.float32).reshape(-1)
            threshold = np.asarray(probe["threshold"], dtype=np.float32).reshape(-1)
            if weights.shape != (bias.size, 4096) or threshold.size != bias.size:
                raise ValueError(f"invalid probe arrays at layer {layer}")
            margins = layers[layer] @ weights.T + bias.reshape(1, -1) - threshold.reshape(1, -1)
            vote_columns.append(margins >= 0.0)
    votes = np.concatenate(vote_columns, axis=1)
    if votes.shape[1] != 225:
        raise ValueError(f"expected 225 V6 votes, found {votes.shape[1]}")
    fractions = votes.mean(axis=1, dtype=np.float64)
    return fractions >= 0.5, fractions


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires a positive sample size")
    p = correct / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - radius, center + radius


def evaluate(labels: np.ndarray, predictions: np.ndarray, fractions: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=bool)
    predictions = np.asarray(predictions, dtype=bool)
    if labels.shape != predictions.shape or labels.shape != fractions.shape:
        raise ValueError("metric inputs have inconsistent shapes")
    tp = int(np.sum(labels & predictions))
    tn = int(np.sum(~labels & ~predictions))
    fp = int(np.sum(~labels & predictions))
    fn = int(np.sum(labels & ~predictions))
    correct = tp + tn
    accuracy = correct / len(labels)
    both_classes = bool(labels.any() and (~labels).any())
    balanced = (tp / (tp + fn) + tn / (tn + fp)) / 2.0 if both_classes else None
    constant_false = float((~labels).mean())
    constant_true = float(labels.mean())
    best_constant = max(constant_false, constant_true)
    lower, upper = wilson_interval(correct, len(labels))
    decision_grade = len(labels) >= 8 and both_classes
    gate_passed = decision_grade and accuracy >= 0.60 and accuracy - best_constant >= 0.10
    return {
        "cases": len(labels),
        "label_counts": {"false": int((~labels).sum()), "true": int(labels.sum())},
        "prediction_counts": {
            "false": int((~predictions).sum()),
            "true": int(predictions.sum()),
        },
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "wilson_95": [lower, upper],
        "always_false_accuracy": constant_false,
        "always_true_accuracy": constant_true,
        "best_constant_accuracy": best_constant,
        "advantage_over_best_constant": accuracy - best_constant,
        "positive_vote_fraction": {
            "min": float(fractions.min()),
            "median": float(np.median(fractions)),
            "mean": float(fractions.mean()),
            "max": float(fractions.max()),
        },
        "decision_grade": decision_grade,
        "gate_passed": gate_passed,
        "decision": "PASS" if gate_passed else ("FAIL" if decision_grade else "INCONCLUSIVE"),
    }


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, str]],
    predictions: np.ndarray,
    fractions: np.ndarray,
    result: dict,
    provenance: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["problem_id", "model_is_robust", "v6_prediction", "positive_vote_fraction"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row, prediction, fraction in zip(rows, predictions, fractions, strict=True):
            writer.writerow(
                {
                    "problem_id": row["problem_id"],
                    "model_is_robust": str(parse_bool(row["model_is_robust"])).lower(),
                    "v6_prediction": str(bool(prediction)).lower(),
                    "positive_vote_fraction": f"{float(fraction):.12f}",
                }
            )
    payload = {"schema_version": 1, "experiment": "E10", **result, "provenance": provenance}
    (output_dir / "validation_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = f"""# E10 independent Small-model OOD result

- Decision: **{result['decision']}**
- Direct Small cases: {result['cases']}
- Labels: {result['label_counts']}
- V6 accuracy: {result['accuracy']:.6f}
- Balanced accuracy: {result['balanced_accuracy'] if result['balanced_accuracy'] is not None else 'not defined'}
- 95% Wilson interval: [{result['wilson_95'][0]:.6f}, {result['wilson_95'][1]:.6f}]
- Always-false / always-true accuracy: {result['always_false_accuracy']:.6f} / {result['always_true_accuracy']:.6f}
- Advantage over best constant: {result['advantage_over_best_constant']:.6f}
- Confusion: {result['confusion']}
- Decision-grade sample: {result['decision_grade']}
- Frozen gate passed: {result['gate_passed']}

E10 is evaluation-only. Individual errors and vote fractions must not be used
for threshold, layer, text-rule, or per-case tuning. A FAIL or INCONCLUSIVE
result cannot authorize V8; V6 remains the frozen Small submission champion.
"""
    (output_dir / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internals-zip", type=Path, required=True)
    parser.add_argument(
        "--v6-zip",
        type=Path,
        default=Path("outputs/aimo-small-v6-fixed-layer-vote-20260910.zip"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/aimo_v6_gpu_ready/validation_e10_ood_small"),
    )
    args = parser.parse_args()
    rows, layers, manifest, audit = load_e10(args.internals_zip)
    labels = np.asarray([parse_bool(row["model_is_robust"]) for row in rows], dtype=bool)
    artifact = load_v6_artifact(args.v6_zip)
    predictions, fractions = predict(artifact, layers)
    result = evaluate(labels, predictions, fractions)
    provenance = {
        "internals_zip": str(args.internals_zip.resolve()),
        "internals_zip_sha256": sha256_file(args.internals_zip),
        "v6_zip": str(args.v6_zip.resolve()),
        "v6_zip_sha256": sha256_file(args.v6_zip),
        "v6_artifact_sha256": V6_ARTIFACT_SHA256,
        "e10_manifest": manifest,
        "e10_dataset_audit": audit,
    }
    write_outputs(args.output_dir, rows, predictions, fractions, result, provenance)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Report: {args.output_dir / 'VALIDATION_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

