#!/usr/bin/env python3
"""Resumable, sharded extraction of input-last-token hidden states.

Each GPU process writes one compressed cache file per problem.  Consolidation is
CPU-only and refuses incomplete or fingerprint-mismatched caches.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("prepared dataset is empty")
    return rows


def build_prompt(tokenizer: Any, user_text: str, system_prompt: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return f"{system_prompt}\n\n{user_text}"


def fingerprint(model_id: str, system_prompt: str, problem: str, max_length: int) -> str:
    payload = json.dumps(
        {
            "format": "input_last_token_v1",
            "model_id": model_id,
            "system_prompt": system_prompt,
            "problem": problem,
            "max_length": max_length,
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def cache_path(cache_dir: Path, row: dict[str, str], digest: str) -> Path:
    safe_id = hashlib.sha256(row["problem_id"].encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{safe_id}-{digest[:16]}.npz"


def cache_is_valid(path: Path, digest: str) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                str(data["fingerprint"].item()) == digest
                and data["hidden_states"].ndim == 2
                and data["hidden_states"].shape[0] >= 2
            )
    except Exception:
        return False


def extract(args: argparse.Namespace) -> None:
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; do not spend hours attempting 8B extraction on CPU")
    rows = read_rows(args.dataset)
    system_prompt = args.system_prompt.read_text(encoding="utf-8").strip()
    work_items = [
        row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index
    ]
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[dict[str, str], str, Path]] = []
    for row in work_items:
        digest = fingerprint(args.model_id, system_prompt, row["original_problem"], args.max_length)
        path = cache_path(args.cache_dir, row, digest)
        if not cache_is_valid(path, digest):
            pending.append((row, digest, path))
    print(
        f"shard {args.shard_index}/{args.num_shards}: {len(work_items)} assigned, "
        f"{len(pending)} pending, {len(work_items) - len(pending)} cached",
        flush=True,
    )
    if not pending:
        return

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.to("cuda")
    model.eval()

    for number, (row, digest, path) in enumerate(pending, start=1):
        prompt = build_prompt(tokenizer, row["original_problem"], system_prompt)
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        token_count = int(encoded["input_ids"].shape[1])
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(
                **encoded,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        if output.hidden_states is None:
            raise RuntimeError("model returned no hidden states")
        matrix = np.stack(
            [state[0, -1, :].detach().float().cpu().numpy() for state in output.hidden_states]
        ).astype(np.float16)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                hidden_states=matrix,
                fingerprint=np.asarray(digest),
                token_count=np.asarray(token_count, dtype=np.int32),
            )
        temporary.replace(path)
        print(
            f"shard {args.shard_index}: [{number}/{len(pending)}] "
            f"{row['problem_id']} tokens={token_count} shape={matrix.shape}",
            flush=True,
        )


def consolidate(args: argparse.Namespace) -> None:
    rows = read_rows(args.dataset)
    system_prompt = args.system_prompt.read_text(encoding="utf-8").strip()
    matrices: list[np.ndarray] = []
    token_counts: list[int] = []
    missing: list[str] = []
    expected_shape: tuple[int, int] | None = None

    for row in rows:
        digest = fingerprint(args.model_id, system_prompt, row["original_problem"], args.max_length)
        path = cache_path(args.cache_dir, row, digest)
        if not cache_is_valid(path, digest):
            missing.append(row["problem_id"])
            continue
        with np.load(path, allow_pickle=False) as data:
            matrix = data["hidden_states"].astype(np.float32)
            token_count = int(data["token_count"].item())
        if expected_shape is None:
            expected_shape = matrix.shape
        elif matrix.shape != expected_shape:
            raise ValueError(
                f"hidden-state shape mismatch for {row['problem_id']}: "
                f"{matrix.shape} != {expected_shape}"
            )
        matrices.append(matrix)
        token_counts.append(token_count)
    if missing:
        raise RuntimeError(f"incomplete extraction: {len(missing)} missing, examples={missing[:10]}")
    if expected_shape is None:
        raise RuntimeError("no valid cache files found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cube = np.stack(matrices)  # [problem, layer, hidden]
    for layer_index in range(cube.shape[1]):
        np.save(args.output_dir / f"layer_{layer_index:03d}.npy", cube[:, layer_index, :])

    with (args.output_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0]) + ["row_index", "token_count", "extraction_view"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (row, token_count) in enumerate(zip(rows, token_counts, strict=True)):
            writer.writerow(
                {
                    **row,
                    "row_index": index,
                    "token_count": token_count,
                    "extraction_view": "input_last_token",
                }
            )

    manifest = {
        "model_id": args.model_id,
        "dataset": str(args.dataset.resolve()),
        "problem_count": len(rows),
        "layer_count": int(cube.shape[1]),
        "hidden_dimension": int(cube.shape[2]),
        "max_token_count": max(token_counts),
        "max_length": args.max_length,
        "representation": "input_last_token",
        "dtype_on_disk": "float32",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="command", required=True)
    for command in ("extract", "consolidate"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--dataset", type=Path, required=True)
        sub.add_argument("--system-prompt", type=Path, required=True)
        sub.add_argument("--model-id", required=True)
        sub.add_argument("--cache-dir", type=Path, required=True)
        sub.add_argument("--max-length", type=int, default=8192)
        if command == "extract":
            sub.add_argument("--num-shards", type=int, default=1)
            sub.add_argument("--shard-index", type=int, default=0)
        else:
            sub.add_argument("--output-dir", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command == "extract":
        extract(args)
    else:
        consolidate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
