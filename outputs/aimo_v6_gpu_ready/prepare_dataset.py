#!/usr/bin/env python3
"""Build the unique-problem public probe dataset and emit a fail-closed audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean label: {value!r}")


def normalized_text(value: str) -> str:
    return " ".join(value.split())


def canonical_problem_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def canonical_content_sha256(rows: list[dict]) -> str:
    payload = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def prepare(input_path: Path, output_path: Path, audit_path: Path) -> dict:
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("input dataset is empty")

    required = {"problem_id", "dataset_id", "original_problem", "model_id", "model_is_robust"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    by_problem: dict[str, dict] = {}
    duplicate_counts: Counter[str] = Counter()
    evaluation_models: set[str] = set()
    for row in rows:
        problem_id = row["problem_id"].strip()
        if not problem_id:
            raise ValueError("blank problem_id")
        current = {
            "problem_id": problem_id,
            "dataset_id": row["dataset_id"].strip(),
            "original_problem": canonical_problem_text(row["original_problem"]),
            "model_is_robust": parse_bool(row["model_is_robust"]),
        }
        evaluation_models.add(row["model_id"].strip())
        if problem_id in by_problem:
            duplicate_counts[problem_id] += 1
            previous = by_problem[problem_id]
            comparable_previous = {
                **previous,
                "original_problem": normalized_text(previous["original_problem"]),
            }
            comparable_current = {
                **current,
                "original_problem": normalized_text(current["original_problem"]),
            }
            if comparable_previous != comparable_current:
                raise ValueError(f"inconsistent duplicate rows for problem_id={problem_id}")
        else:
            by_problem[problem_id] = current

    if len(evaluation_models) != 1:
        raise ValueError(f"expected one evaluation model, found {sorted(evaluation_models)}")

    unique_rows = [by_problem[key] for key in sorted(by_problem)]
    text_to_ids: defaultdict[str, list[str]] = defaultdict(list)
    for row in unique_rows:
        text_to_ids[normalized_text(row["original_problem"])].append(row["problem_id"])
    text_collisions = {text: ids for text, ids in text_to_ids.items() if len(ids) > 1}
    if text_collisions:
        examples = list(text_collisions.values())[:5]
        raise ValueError(f"same normalized prompt has multiple problem_ids: {examples}")

    by_source = Counter(row["dataset_id"] for row in unique_rows)
    by_label = Counter(str(row["model_is_robust"]) for row in unique_rows)
    source_label: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in unique_rows:
        source_label[row["dataset_id"]][str(row["model_is_robust"])] += 1
    if len(by_source) < 2:
        raise ValueError("source-holdout validation requires at least two sources")
    if any(set(counts) != {"True", "False"} for counts in source_label.values()):
        raise ValueError("each source must contain both labels")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["problem_id", "dataset_id", "original_problem", "model_is_robust"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in unique_rows:
            writer.writerow({**row, "model_is_robust": str(row["model_is_robust"]).lower()})

    digest = hashlib.sha256(output_path.read_bytes()).hexdigest().upper()
    audit = {
        "input": str(input_path.resolve()),
        "input_rows": len(rows),
        "unique_problem_rows": len(unique_rows),
        "removed_duplicate_rows": len(rows) - len(unique_rows),
        "duplicated_problem_ids": dict(sorted(duplicate_counts.items())),
        "evaluation_model_id": next(iter(evaluation_models)),
        "counts_by_source": dict(sorted(by_source.items())),
        "counts_by_label": dict(sorted(by_label.items())),
        "counts_by_source_and_label": {
            source: dict(sorted(counts.items())) for source, counts in sorted(source_label.items())
        },
        "prepared_sha256": digest,
        "canonical_content_sha256": canonical_content_sha256(unique_rows),
        "grouping_key": "problem_id",
        "leakage_checks_passed": True,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    audit = prepare(args.input, args.output, args.audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
