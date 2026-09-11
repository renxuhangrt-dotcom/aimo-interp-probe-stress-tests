#!/usr/bin/env python3
"""Build the technical-report table, figure, and reproducibility manifest."""

from __future__ import annotations

import csv
import hashlib
import json
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parent
VALIDATION = OUTPUTS / "aimo_v6_gpu_ready"

EXPERIMENTS = [
    ("V6", "validation_e2_layer_vote", "Final prompt token + linear vote"),
    ("E6", "validation_e6_layer_delta_vote", "Adjacent-layer deltas + linear vote"),
    ("E7", "validation_e7_problem_mean", "Problem-token mean + linear vote"),
    ("E8", "validation_e8_problem_last", "Final problem token + linear vote"),
    ("E9", "validation_e9_pca_rbf", "Final prompt token + PCA-RBF vote"),
]


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def load_rows() -> tuple[list[dict[str, object]], dict]:
    rows: list[dict[str, object]] = []
    for label, directory, method in EXPERIMENTS:
        result = read_json(VALIDATION / directory / "validation_result.json")
        metrics = result["metrics"]
        holdouts = list(metrics["source_holdout_balanced_accuracy"].values())
        if len(holdouts) != 2:
            raise ValueError(f"{label} must have exactly two source holdouts")
        rows.append(
            {
                "experiment": label,
                "method": method,
                "grouped_oof_balanced_accuracy": metrics["grouped_balanced_accuracy"],
                "grouped_oof_accuracy": metrics["ordinary_accuracy"],
                "delta_balanced_accuracy_vs_v6": metrics.get(
                    "balanced_accuracy_difference_vs_v6", 0.0
                ),
                "bootstrap_95_lower": metrics["bootstrap_95_lower"],
                "source_holdout_a_balanced_accuracy": holdouts[0],
                "source_holdout_b_balanced_accuracy": holdouts[1],
                "source_holdout_mean_balanced_accuracy": sum(holdouts) / len(holdouts),
                "randomized_control_advantage": metrics["randomized_control_advantage"],
                "disagreement_vs_v6": metrics.get("disagreement_vs_v6"),
                "gate_passed": bool(result["gate"]["passed"]),
            }
        )
    e10 = read_json(VALIDATION / "validation_e10_ood_small" / "validation_result.json")
    return rows, e10


def write_table(rows: list[dict[str, object]], e10: dict) -> None:
    fieldnames = [
        "experiment",
        "method",
        "grouped_oof_balanced_accuracy",
        "grouped_oof_accuracy",
        "delta_balanced_accuracy_vs_v6",
        "bootstrap_95_lower",
        "source_holdout_a_balanced_accuracy",
        "source_holdout_b_balanced_accuracy",
        "source_holdout_mean_balanced_accuracy",
        "randomized_control_advantage",
        "disagreement_vs_v6",
        "gate_passed",
    ]
    with (HERE / "results_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: fmt(value) if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )
    with (HERE / "e10_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cases",
                "negative_labels",
                "positive_labels",
                "v6_accuracy",
                "v6_balanced_accuracy",
                "best_constant_accuracy",
                "advantage_over_best_constant",
                "predicted_positive",
                "decision",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "cases": e10["cases"],
                "negative_labels": e10["label_counts"]["false"],
                "positive_labels": e10["label_counts"]["true"],
                "v6_accuracy": fmt(e10["accuracy"]),
                "v6_balanced_accuracy": fmt(e10["balanced_accuracy"]),
                "best_constant_accuracy": fmt(e10["best_constant_accuracy"]),
                "advantage_over_best_constant": fmt(e10["advantage_over_best_constant"]),
                "predicted_positive": e10["prediction_counts"]["true"],
                "decision": e10["decision"],
            }
        )


def write_figure(rows: list[dict[str, object]], e10: dict) -> None:
    width, height = 1120, 470
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111827}.title{font-size:18px;font-weight:700}.panel{font-size:15px;font-weight:700}.tick{font-size:11px;fill:#475569}.label{font-size:12px}.note{font-size:11px;fill:#475569}.value{font-size:11px;font-weight:700}</style>',
        '<text x="560" y="27" text-anchor="middle" class="title">Small in-distribution gains did not transfer to a new problem source</text>',
    ]

    # Panel A: grouped public balanced accuracy.
    left, top, plot_w, plot_h = 70, 80, 500, 300
    ymin, ymax = 0.55, 0.76

    def y_a(value: float) -> float:
        return top + plot_h * (ymax - value) / (ymax - ymin)

    svg.append('<text x="70" y="58" class="panel">A. Grouped public validation</text>')
    for tick in [0.55, 0.60, 0.65, 0.70, 0.75]:
        y = y_a(tick)
        svg.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e2e8f0"/>',
                f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{tick:.2f}</text>',
            ]
        )
    svg.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#64748b"/>',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#64748b"/>',
            f'<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})" text-anchor="middle" class="label">Balanced accuracy</text>',
        ]
    )
    bar_w = 58
    gap = plot_w / len(rows)
    for index, row in enumerate(rows):
        x = left + gap * (index + 0.5)
        score = float(row["grouped_oof_balanced_accuracy"])
        lower = float(row["bootstrap_95_lower"])
        y_score = y_a(score)
        y_base = y_a(ymin)
        y_lower = y_a(lower)
        color = "#2563eb" if index == 0 else "#94a3b8"
        svg.extend(
            [
                f'<rect x="{x - bar_w / 2:.1f}" y="{y_score:.1f}" width="{bar_w}" height="{y_base - y_score:.1f}" fill="{color}"/>',
                f'<line x1="{x:.1f}" y1="{y_score:.1f}" x2="{x:.1f}" y2="{y_lower:.1f}" stroke="#111827" stroke-width="1.5"/>',
                f'<line x1="{x - 7:.1f}" y1="{y_lower:.1f}" x2="{x + 7:.1f}" y2="{y_lower:.1f}" stroke="#111827" stroke-width="1.5"/>',
                f'<text x="{x:.1f}" y="{y_score - 7:.1f}" text-anchor="middle" class="value">{score:.3f}</text>',
                f'<text x="{x:.1f}" y="{top + plot_h + 19}" text-anchor="middle" class="label">{escape(str(row["experiment"]))}</text>',
            ]
        )
    v6_score = float(rows[0]["grouped_oof_balanced_accuracy"])
    for value, color, dash, text_value in [
        (v6_score, "#2563eb", "6 4", "V6"),
        (v6_score + 0.02, "#dc2626", "2 4", "promotion gate"),
    ]:
        y = y_a(value)
        svg.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="{color}" stroke-dasharray="{dash}"/>',
                f'<text x="{left + plot_w - 4}" y="{y - 4:.1f}" text-anchor="end" class="note" fill="{color}">{text_value}</text>',
            ]
        )
    svg.append('<text x="75" y="430" class="note">Vertical marks end at one-sided 95% bootstrap lower bounds.</text>')

    # Panel B: independent OOD ordinary accuracy.
    left_b, plot_w_b = 670, 380
    svg.append(f'<text x="{left_b}" y="58" class="panel">B. Independent OOD audit (n={int(e10["cases"])})</text>')

    def y_b(value: float) -> float:
        return top + plot_h * (1.0 - value)

    for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = y_b(tick)
        svg.extend(
            [
                f'<line x1="{left_b}" y1="{y:.1f}" x2="{left_b + plot_w_b}" y2="{y:.1f}" stroke="#e2e8f0"/>',
                f'<text x="{left_b - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{tick:.1f}</text>',
            ]
        )
    svg.extend(
        [
            f'<line x1="{left_b}" y1="{top}" x2="{left_b}" y2="{top + plot_h}" stroke="#64748b"/>',
            f'<line x1="{left_b}" y1="{top + plot_h}" x2="{left_b + plot_w_b}" y2="{top + plot_h}" stroke="#64748b"/>',
            f'<text x="620" y="{top + plot_h / 2}" transform="rotate(-90 620 {top + plot_h / 2})" text-anchor="middle" class="label">Accuracy</text>',
        ]
    )
    values = [float(e10["accuracy"]), float(e10["best_constant_accuracy"])]
    names = ["Frozen V6", "Always negative"]
    colors_b = ["#2563eb", "#f59e0b"]
    centers = [left_b + 115, left_b + 280]
    for x, value, name, color in zip(centers, values, names, colors_b, strict=True):
        y = y_b(value)
        svg.extend(
            [
                f'<rect x="{x - 42}" y="{y:.1f}" width="84" height="{top + plot_h - y:.1f}" fill="{color}"/>',
                f'<text x="{x}" y="{y - 8:.1f}" text-anchor="middle" class="value">{value:.2f}</text>',
                f'<text x="{x}" y="{top + plot_h + 19}" text-anchor="middle" class="label">{name}</text>',
            ]
        )
    lower, upper = [float(value) for value in e10["wilson_95"]]
    x = centers[0]
    svg.extend(
        [
            f'<line x1="{x}" y1="{y_b(upper):.1f}" x2="{x}" y2="{y_b(lower):.1f}" stroke="#111827" stroke-width="1.5"/>',
            f'<line x1="{x - 7}" y1="{y_b(upper):.1f}" x2="{x + 7}" y2="{y_b(upper):.1f}" stroke="#111827" stroke-width="1.5"/>',
            f'<line x1="{x - 7}" y1="{y_b(lower):.1f}" x2="{x + 7}" y2="{y_b(lower):.1f}" stroke="#111827" stroke-width="1.5"/>',
            f'<text x="{left_b + plot_w_b / 2}" y="96" text-anchor="middle" class="note" fill="#991b1b">V6 predicted all 10 cases robust</text>',
        ]
    )
    svg.append('</svg>')
    (HERE / "results_overview.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")


def write_manifest() -> None:
    paths = [
        OUTPUTS / "aimo-small-v6-fixed-layer-vote-20260910.zip",
        OUTPUTS / "EXPERIMENT_LOG.md",
        OUTPUTS / "OFFICIAL_DATA_AUDIT_AND_E10_PLAN.md",
        VALIDATION / "E6_PREREGISTRATION.md",
        VALIDATION / "E7_PREREGISTRATION.md",
        VALIDATION / "E8_PREREGISTRATION.md",
        VALIDATION / "E9_PREREGISTRATION.md",
        VALIDATION / "E10_PREREGISTRATION.md",
    ]
    for _, directory, _ in EXPERIMENTS:
        paths.append(VALIDATION / directory / "validation_result.json")
    paths.append(VALIDATION / "validation_e10_ood_small" / "validation_result.json")
    paths.extend(
        [
            HERE / "TECHNICAL_REPORT_DRAFT.md",
            HERE / "TECHNICAL_REPORT_2PAGE_DRAFT.md",
            HERE / "NEW_MODEL_GATE.md",
            HERE / "ROADMAP_ZH.md",
            HERE / "SUBMISSION_CHECKLIST.md",
            HERE / "build_report_assets.py",
            HERE / "build_two_page_pdf.py",
            HERE / "test_report_integrity.py",
            HERE / "results_summary.csv",
            HERE / "e10_summary.csv",
            HERE / "results_overview.svg",
            HERE / "results_overview.png",
            HERE / "pdf" / "When_Good_Probes_Fail_Xuhang_Ren_submission.pdf",
        ]
    )
    manifest = {
        "schema_version": 1,
        "generated_from_frozen_results": True,
        "files": [
            {
                "path": path.relative_to(OUTPUTS).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in paths
        ],
    }
    (HERE / "reproducibility_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    rows, e10 = load_rows()
    write_table(rows, e10)
    write_figure(rows, e10)
    write_manifest()
    print(f"Wrote report assets to {HERE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
