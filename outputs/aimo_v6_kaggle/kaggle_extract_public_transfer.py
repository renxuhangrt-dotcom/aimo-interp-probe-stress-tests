# %% [markdown]
# # AIMO — public cross-model transfer extraction on Kaggle T4 x2
#
# Select **GPU T4 x2** and enable Internet before running.  This script extracts
# only; it never creates or submits V6.  Results are written under
# `/kaggle/working/aimo_v6_kaggle/` and survive reruns as notebook outputs.

# %%
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


PINNED_BASELINE_COMMIT = "7e8839966750059b6b1d247a12ab56552c79b342"
NOTEBOOK_VERSION = "public-transfer-1"
BASELINE_REPOSITORY = "https://github.com/aimo-interp/baselines.git"
MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
MAX_LENGTH = 8192
WORKING_ROOT = Path("/kaggle/working/aimo_public_transfer")
TEMP_ROOT = Path("/kaggle/temp/aimo_public_transfer")
REPOSITORY_DIR = TEMP_ROOT / "official-baselines"
MODEL_CACHE = TEMP_ROOT / "huggingface"
CACHE_DIR = WORKING_ROOT / "problem_cache"
INTERNALS_DIR = WORKING_ROOT / "internals"
PREPARED_DATASET = WORKING_ROOT / "problems.csv"
DATASET_AUDIT = WORKING_ROOT / "dataset_audit.json"
PROGRESS_PATH = WORKING_ROOT / "progress.json"
FINAL_ZIP = Path("/kaggle/working/aimo_public_transfer_internals.zip")
PUBLIC_DATASET_URL = (
    "https://huggingface.co/datasets/aimo-interp/val-sample/resolve/main/"
    "data/validation-00000-of-00001.parquet"
)
PUBLIC_DATASET_SHA256 = "3919CB3D951BB360A120BE3397E926F48B7819BCD97E12CFEE4C01E76618560F"


def run(command: list[str], **kwargs) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def ensure_runtime_packages() -> None:
    """Install only missing/old HF libraries; never replace Kaggle's CUDA torch."""
    requirements = {"transformers": "4.51", "accelerate": "0.30"}
    missing = []
    for package, minimum in requirements.items():
        try:
            current = importlib.metadata.version(package)
            current_tuple = tuple(int(part) for part in current.split(".")[:2])
            minimum_tuple = tuple(int(part) for part in minimum.split("."))
            if current_tuple < minimum_tuple:
                missing.append(f"{package}>={minimum}")
        except importlib.metadata.PackageNotFoundError:
            missing.append(f"{package}>={minimum}")
    if missing:
        run([sys.executable, "-m", "pip", "install", "--quiet", *missing])


def gpu_preflight() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device. In Kaggle Settings choose Accelerator: GPU T4 x2.")
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": props.name,
                "memory_gib": round(props.total_memory / 1024**3, 2),
                "capability": f"{props.major}.{props.minor}",
            }
        )
    if len(devices) != 2:
        raise RuntimeError(
            f"This exact-weight workflow requires two GPUs; Kaggle exposed {len(devices)}: {devices}"
        )
    if any("P100" in device["name"].upper() for device in devices):
        raise RuntimeError("P100 is intentionally refused; select Kaggle T4 x2 instead.")
    if any(device["memory_gib"] < 14.0 for device in devices):
        raise RuntimeError(f"Each GPU must have at least 14 GiB: {devices}")
    print(json.dumps({"torch": torch.__version__, "devices": devices}, indent=2), flush=True)
    return {"torch": torch.__version__, "devices": devices}


def checkout_official_baseline() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    if not (REPOSITORY_DIR / ".git").exists():
        run(["git", "clone", "--filter=blob:none", BASELINE_REPOSITORY, str(REPOSITORY_DIR)])
    run(["git", "-C", str(REPOSITORY_DIR), "fetch", "--quiet", "origin"])
    run(["git", "-C", str(REPOSITORY_DIR), "checkout", "--quiet", "--detach", PINNED_BASELINE_COMMIT])
    actual = subprocess.check_output(
        ["git", "-C", str(REPOSITORY_DIR), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != PINNED_BASELINE_COMMIT:
        raise RuntimeError(f"baseline commit mismatch: {actual}")


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


def prepare_dataset() -> tuple[list[dict[str, str]], dict]:
    input_path = REPOSITORY_DIR / "data" / "math-robust-agg.csv"
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        input_rows = list(csv.DictReader(handle))
    required = {"problem_id", "dataset_id", "original_problem", "model_id", "model_is_robust"}
    if not input_rows or required.difference(input_rows[0]):
        raise ValueError("official aggregate dataset is empty or has an unexpected schema")

    by_problem: dict[str, dict] = {}
    duplicate_counts: Counter[str] = Counter()
    evaluation_models: set[str] = set()
    for row in input_rows:
        problem_id = row["problem_id"].strip()
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
            if (
                previous["dataset_id"] != current["dataset_id"]
                or previous["model_is_robust"] != current["model_is_robust"]
                or normalized_text(previous["original_problem"])
                != normalized_text(current["original_problem"])
            ):
                raise ValueError(f"inconsistent duplicate rows for {problem_id}")
        else:
            by_problem[problem_id] = current
    if evaluation_models != {"qwen3-8b:low"}:
        raise ValueError(f"unexpected evaluation model labels: {sorted(evaluation_models)}")

    rows = [by_problem[key] for key in sorted(by_problem)]
    text_to_id: dict[str, str] = {}
    for row in rows:
        text = normalized_text(row["original_problem"])
        if text in text_to_id and text_to_id[text] != row["problem_id"]:
            raise ValueError("same normalized prompt appears under multiple problem IDs")
        text_to_id[text] = row["problem_id"]

    WORKING_ROOT.mkdir(parents=True, exist_ok=True)
    with PREPARED_DATASET.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["problem_id", "dataset_id", "original_problem", "model_is_robust"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "model_is_robust": str(row["model_is_robust"]).lower()})

    source_counts = Counter(row["dataset_id"] for row in rows)
    label_counts = Counter(str(row["model_is_robust"]) for row in rows)
    source_labels: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        source_labels[row["dataset_id"]][str(row["model_is_robust"])] += 1
    audit = {
        "official_commit": PINNED_BASELINE_COMMIT,
        "input_rows": len(input_rows),
        "unique_problem_rows": len(rows),
        "removed_duplicate_rows": len(input_rows) - len(rows),
        "duplicated_problem_ids": dict(sorted(duplicate_counts.items())),
        "counts_by_source": dict(sorted(source_counts.items())),
        "counts_by_label": dict(sorted(label_counts.items())),
        "counts_by_source_and_label": {
            key: dict(sorted(value.items())) for key, value in sorted(source_labels.items())
        },
        "prepared_file_sha256": hashlib.sha256(PREPARED_DATASET.read_bytes()).hexdigest().upper(),
        "canonical_content_sha256": canonical_content_sha256(rows),
        "leakage_checks_passed": True,
    }
    if len(rows) != 137 or audit["canonical_content_sha256"] != (
        "3FC14BFC02F7C1D0D79BB917491C7710F6A798150416B157C6C34205CEC9E067"
    ):
        raise RuntimeError(f"prepared dataset does not match the preregistered dataset: {audit}")
    DATASET_AUDIT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    return rows, audit


def prepare_public_transfer_dataset() -> tuple[list[dict[str, str]], dict]:
    """Download the frozen public sample and keep one row per unique problem."""
    import pandas as pd

    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    source_path = TEMP_ROOT / "validation-00000-of-00001.parquet"
    urllib.request.urlretrieve(PUBLIC_DATASET_URL, source_path)
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest().upper()
    if source_hash != PUBLIC_DATASET_SHA256:
        raise RuntimeError(
            f"public sample hash changed: {source_hash} != {PUBLIC_DATASET_SHA256}"
        )
    frame = pd.read_parquet(source_path)
    required = {"original_problem"}
    if frame.empty or required.difference(frame.columns):
        raise RuntimeError(f"unexpected public sample columns: {sorted(frame.columns)}")

    unique: dict[str, dict[str, str]] = {}
    for value in frame["original_problem"].tolist():
        problem = canonical_problem_text(str(value))
        normalized = normalized_text(problem)
        if normalized not in unique:
            digest = hashlib.sha256(problem.encode("utf-8")).hexdigest()[:12]
            unique[normalized] = {
                "problem_id": f"public-{digest}",
                "dataset_id": "aimo-interp/val-sample",
                "original_problem": problem,
                # Placeholder only: transfer scoring uses the separate public
                # per-model labels after these representations are downloaded.
                "model_is_robust": False,
            }
    rows = sorted(unique.values(), key=lambda row: row["problem_id"])
    if len(frame) != 28 or len(rows) != 8:
        raise RuntimeError(
            f"public sample cardinality changed: rows={len(frame)}, unique_problems={len(rows)}"
        )

    WORKING_ROOT.mkdir(parents=True, exist_ok=True)
    with PREPARED_DATASET.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["problem_id", "dataset_id", "original_problem", "model_is_robust"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "model_is_robust": "false"})
    audit = {
        "source": PUBLIC_DATASET_URL,
        "source_file_sha256": source_hash,
        "input_rows": len(frame),
        "unique_problem_rows": len(rows),
        "prepared_file_sha256": hashlib.sha256(PREPARED_DATASET.read_bytes()).hexdigest().upper(),
        "canonical_content_sha256": canonical_content_sha256(rows),
        "labels_used_for_extraction": False,
        "leakage_checks_passed": True,
    }
    DATASET_AUDIT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    return rows, audit


def build_prompt(tokenizer, user_text: str, system_prompt: str) -> str:
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


def fingerprint(system_prompt: str, problem: str) -> str:
    payload = json.dumps(
        {
            "format": "input_last_token_v1",
            "model_id": MODEL_ID,
            "system_prompt": system_prompt,
            "problem": problem,
            "max_length": MAX_LENGTH,
            "weight_dtype": "float16",
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def cache_path(row: dict[str, str], digest: str) -> Path:
    safe_id = hashlib.sha256(row["problem_id"].encode()).hexdigest()[:12]
    return CACHE_DIR / f"{safe_id}-{digest[:16]}.npz"


def cache_valid(path: Path, digest: str) -> bool:
    import numpy as np

    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                str(data["fingerprint"].item()) == digest
                and data["hidden_states"].ndim == 2
                and data["hidden_states"].shape == (37, 4096)
            )
    except Exception:
        return False


def import_previous_cache() -> int:
    """Reuse cache files when a previous notebook version is attached as Input."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return 0
    imported = 0
    for source in input_root.glob("**/problem_cache/*.npz"):
        destination = CACHE_DIR / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
            imported += 1
    if imported:
        print(f"Imported {imported} cache files from attached previous output", flush=True)
    return imported


def model_devices(model) -> dict:
    mapping = getattr(model, "hf_device_map", None)
    if not isinstance(mapping, dict):
        raise RuntimeError("Transformers did not create a multi-GPU device map")
    placements = {str(value) for value in mapping.values()}
    if "cpu" in placements or "disk" in placements:
        raise RuntimeError(f"CPU/disk offload is refused; model placements={sorted(placements)}")
    gpu_ids = set()
    for value in mapping.values():
        text = str(value)
        if text.startswith("cuda:"):
            gpu_ids.add(int(text.split(":", 1)[1]))
        elif text.isdigit():
            gpu_ids.add(int(text))
        elif isinstance(value, int):
            gpu_ids.add(value)
    if gpu_ids != {0, 1}:
        raise RuntimeError(f"model must span both GPUs; device map uses {sorted(gpu_ids)}")
    return {str(key): str(value) for key, value in mapping.items()}


def save_progress(
    rows: list[dict[str, str]],
    system_prompt: str,
    status: str,
    completed: int | None = None,
) -> dict:
    if completed is None:
        completed = 0
        for row in rows:
            digest = fingerprint(system_prompt, row["original_problem"])
            completed += int(cache_valid(cache_path(row, digest), digest))
    progress = {
        "status": status,
        "completed": completed,
        "total": len(rows),
        "remaining": len(rows) - completed,
        "updated_unix": time.time(),
    }
    PROGRESS_PATH.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
    return progress


def extract(rows: list[dict[str, str]], hardware: dict) -> tuple[str, dict]:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    import_previous_cache()
    system_prompt = (REPOSITORY_DIR / "prompts" / "solve.txt").read_text(encoding="utf-8").strip()
    pending = []
    for row in rows:
        digest = fingerprint(system_prompt, row["original_problem"])
        path = cache_path(row, digest)
        if not cache_valid(path, digest):
            pending.append((row, digest, path))
    print(f"Extraction: {len(rows) - len(pending)}/{len(rows)} cached; {len(pending)} pending", flush=True)
    initial_completed = len(rows) - len(pending)
    if not pending:
        saved_map = WORKING_ROOT / "model_device_map.json"
        return system_prompt, {
            "reused_all_cache": True,
            "hardware": hardware,
            "model_device_map": json.loads(saved_map.read_text()) if saved_map.exists() else None,
        }

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=MODEL_CACHE,
        torch_dtype=torch.float16,
        device_map="balanced",
        max_memory={0: "13GiB", 1: "13GiB", "cpu": "20GiB"},
        low_cpu_mem_usage=True,
    )
    placements = model_devices(model)
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    print(f"Input device: {input_device}; mapped modules: {len(placements)}", flush=True)
    (WORKING_ROOT / "model_device_map.json").write_text(
        json.dumps(placements, indent=2) + "\n", encoding="utf-8"
    )

    started = time.time()
    for number, (row, digest, path) in enumerate(pending, start=1):
        prompt = build_prompt(tokenizer, row["original_problem"], system_prompt)
        encoded = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
        )
        token_count = int(encoded["input_ids"].shape[1])
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(
                **encoded,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        if output.hidden_states is None or len(output.hidden_states) != 37:
            raise RuntimeError("expected embedding output plus 36 transformer-layer hidden states")
        matrix = np.stack(
            [state[0, -1, :].detach().float().cpu().numpy() for state in output.hidden_states]
        ).astype(np.float16)
        if matrix.shape != (37, 4096) or not np.isfinite(matrix).all():
            raise RuntimeError(f"invalid hidden-state matrix for {row['problem_id']}: {matrix.shape}")
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                hidden_states=matrix,
                fingerprint=np.asarray(digest),
                token_count=np.asarray(token_count, dtype=np.int32),
            )
        temporary.replace(path)
        del output, matrix, encoded
        progress = save_progress(
            rows, system_prompt, "extracting", completed=initial_completed + number
        )
        elapsed = time.time() - started
        rate = elapsed / number
        print(
            f"[{number}/{len(pending)}] {row['problem_id']} tokens={token_count}; "
            f"overall={progress['completed']}/{progress['total']}; "
            f"mean={rate:.1f}s/problem",
            flush=True,
        )
    del model, tokenizer
    torch.cuda.empty_cache()
    return system_prompt, {
        "reused_all_cache": False,
        "hardware": hardware,
        "model_device_map": placements,
    }


def consolidate(rows: list[dict[str, str]], system_prompt: str, extraction: dict, audit: dict) -> dict:
    import numpy as np

    matrices = []
    token_counts = []
    missing = []
    for row in rows:
        digest = fingerprint(system_prompt, row["original_problem"])
        path = cache_path(row, digest)
        if not cache_valid(path, digest):
            missing.append(row["problem_id"])
            continue
        with np.load(path, allow_pickle=False) as data:
            matrices.append(data["hidden_states"].astype(np.float32))
            token_counts.append(int(data["token_count"].item()))
    if missing:
        save_progress(rows, system_prompt, "incomplete")
        raise RuntimeError(f"extraction incomplete; missing {len(missing)}: {missing[:10]}")

    if INTERNALS_DIR.exists():
        shutil.rmtree(INTERNALS_DIR)
    INTERNALS_DIR.mkdir(parents=True)
    cube = np.stack(matrices)
    for layer_index in range(cube.shape[1]):
        np.save(INTERNALS_DIR / f"layer_{layer_index:03d}.npy", cube[:, layer_index, :])
    with (INTERNALS_DIR / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0]) + ["row_index", "token_count", "extraction_view"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (row, count) in enumerate(zip(rows, token_counts, strict=True)):
            writer.writerow(
                {**row, "row_index": index, "token_count": count, "extraction_view": "input_last_token"}
            )
    manifest = {
        "schema_version": 1,
        "notebook_version": NOTEBOOK_VERSION,
        "official_commit": PINNED_BASELINE_COMMIT,
        "model_id": MODEL_ID,
        "weight_dtype": "float16",
        "representation": "input_last_token",
        "problem_count": len(rows),
        "layer_count": int(cube.shape[1]),
        "hidden_dimension": int(cube.shape[2]),
        "max_token_count": max(token_counts),
        "prepared_dataset_file_sha256": audit["prepared_file_sha256"],
        "prepared_dataset_canonical_sha256": audit["canonical_content_sha256"],
        "extraction": extraction,
    }
    (INTERNALS_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    save_progress(rows, system_prompt, "complete", completed=len(rows))
    shutil.copy2(DATASET_AUDIT, INTERNALS_DIR / "dataset_audit.json")
    shutil.copy2(PROGRESS_PATH, INTERNALS_DIR / "progress.json")
    saved_map = WORKING_ROOT / "model_device_map.json"
    if saved_map.exists():
        shutil.copy2(saved_map, INTERNALS_DIR / "model_device_map.json")
    if FINAL_ZIP.exists():
        FINAL_ZIP.unlink()
    archive_base = str(FINAL_ZIP.with_suffix(""))
    shutil.make_archive(archive_base, "zip", root_dir=WORKING_ROOT, base_dir="internals")
    manifest["result_zip"] = str(FINAL_ZIP)
    manifest["result_zip_sha256"] = hashlib.sha256(FINAL_ZIP.read_bytes()).hexdigest().upper()
    (WORKING_ROOT / "FINAL_RESULT.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main() -> None:
    os.environ["HF_HOME"] = str(MODEL_CACHE)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ensure_runtime_packages()
    hardware = gpu_preflight()
    checkout_official_baseline()
    rows, audit = prepare_public_transfer_dataset()
    system_prompt, extraction = extract(rows, hardware)
    manifest = consolidate(rows, system_prompt, extraction, audit)
    print("\nSUCCESS")
    print(f"Download this file from Kaggle Output: {manifest['result_zip']}")
    print(f"SHA-256: {manifest['result_zip_sha256']}")


if __name__ == "__main__":
    main()
