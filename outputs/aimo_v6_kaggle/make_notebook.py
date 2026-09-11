#!/usr/bin/env python3
"""Convert the percent-format Kaggle script into a one-file .ipynb artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert(source_path: Path, output_path: Path) -> None:
    source = source_path.read_text(encoding="utf-8")
    cells = []
    kind = "code"
    lines: list[str] = []

    def flush() -> None:
        nonlocal lines
        if not lines:
            return
        if kind == "markdown":
            cleaned = []
            for line in lines:
                if line.startswith("# "):
                    cleaned.append(line[2:])
                elif line.startswith("#"):
                    cleaned.append(line[1:].lstrip())
                else:
                    cleaned.append(line)
            cells.append({"cell_type": "markdown", "metadata": {}, "source": cleaned})
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": lines,
                }
            )
        lines = []

    for line in source.splitlines(keepends=True):
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
        else:
            lines.append(line)
    flush()
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert(args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
