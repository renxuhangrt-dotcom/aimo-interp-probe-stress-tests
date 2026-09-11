# %% [markdown]
# # AIMO E12 — counterbalanced metacognitive robustness readout
#
# Select **GPU T4 x2** and enable Internet.  This notebook is a fail-closed
# experiment, not a submission builder. Stage A touches only the public 137
# rows. External features and labels remain untouched unless Stage A passes.

# %%
from __future__ import annotations

import csv
import base64
import hashlib
import importlib.metadata
import io
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import warnings
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path


NOTEBOOK_VERSION = "e12-counterbalanced-metacognitive-readout-1"
OFFICIAL_COMMIT = "7e8839966750059b6b1d247a12ab56552c79b342"
OFFICIAL_CANONICAL_SHA256 = "3FC14BFC02F7C1D0D79BB917491C7710F6A798150416B157C6C34205CEC9E067"
OFFICIAL_REPOSITORY = "https://github.com/aimo-interp/baselines.git"
MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
MODEL_REVISION = "6e8885a6ff5c1dc5201574c8fd700323f23c25fa"
AIME_DATASET_ID = "HuggingFaceH4/aime_2024"
AIME_REVISION = "2fe88a2f1091d5048c0f36abc874fb997b3dd99a"
RRB_REPOSITORY = "https://github.com/pavelgolikov/Robust-Reasoning-Benchmark"
RRB_COMMIT = "0ae533b7bda703747089ffd1b5574fc010ea05f7"

LAYERS = [4, 8, 12, 16, 20, 24, 28, 32, 36]
DIAGNOSTIC_VIEWS = [
    "robustness_stable_a",
    "robustness_stable_b",
    "correctness_correct_a",
    "correctness_correct_b",
]
RRB_VARIANTS = ["baseline", "sentence_reversal", "word_reversal", "split_reversal"]
SEEDS = [42, 43, 44, 45, 46]
FOLDS = 5
V6_C = 0.001
CANDIDATE_C = 0.1
V6_REFERENCE_BA = 0.7047579757975797
V6_REFERENCE_LOWER = 0.6169437631511935
V6_PUBLIC_REPOSITORY_COMMIT = "c87824100d0570078efb21f564ea885bd4acd017"
V6_PUBLIC_ZIP_URL = (
    "https://raw.githubusercontent.com/renxuhangrt-dotcom/"
    "aimo-interp-probe-stress-tests/"
    f"{V6_PUBLIC_REPOSITORY_COMMIT}/outputs/"
    "aimo-small-v6-fixed-layer-vote-20260910.zip"
)
V6_PUBLIC_ZIP_SHA256 = "C757708FCD5F079AF7A90EC30E91D45E5E9542FDBA32C125A862ACC68D933F3E"
V6_ARTIFACT_SHA256 = "7A4DF77D031530DC4998CBC4B6FFF2B8B717F99C7FA04E2646847C6A5BE3672B"
V6_OOF_CSV_SHA256 = "6AE22FC2C40CAF339F0E1F79452B074F07B0296114B9DF47F0629179DB67B6D7"
V6_OOF_CSV_ZLIB_BASE64 = (
    "eNqtWdtuHDcMfc+vdG1Q1P0H+gV9NyRSQow6trHeBujfV84Y8UxnFQyZ1YsXtsUlpcPDQ+r1/FKf2reHRz5xuZS3dnn/+FRqezq9vrw9Xh6/t4fvL5f20M+FLo8vz6fXc+PH5SO9nM+NLl/AQ6x0+tqe+fwv/f12961cvt5dzuXx+e7ceju3Z2p3HuD0Z3l6aye4T58rn/46/9OWv3wBdhaaxFL2P1fYWuJeu8DSx893E1+g9dzykc0/vhHuA36u5XeLnQ7AURKOXYez9qg3QLOz1N4uK0NmZQhw7dLK0vin1lHik48/V1ofsYkxyC7Lm8+1sdTLtcvaR/dx3PtL/xEaAlKtEodmh4TGUU8SSxvwIVbu9ng8YZcOixfO9sRKDGPoSCxAjJlADxN5Jomh9S2vDVEtMuRNXerZo4BwNmCzEAqJ7hY2dLXywyJ3AyKYfK64ccpjAkl+p5mhEBHhMHWZFdxsIGv8YR+2hxpTtcf3bpBqyYPge0f628+1cYJq6FkLixYSW2Wy2W4SZfXm2H1Rbnb3of1hHCijdp7sFcd/cfaTYuBKblFW6iY04WqPOQlcihMJ4ChfY+GD5zrkQ3DHKdyuMbnYWbzwQGCbKiG9KYmc8mY9DlBGXTZ7ZJbolv9tbiwpPOtT9zY5kaTYbHZYQETsMyLxEU2W1M6w3T2wU45DZ0/lSziDFWrRAacMPRNvkNW+2qG1BELYrdY2DWoaIu/3JbVvljAqERKgh1gE5zLjqGBDBhjKQCQjNsV/bW2IGVHpmVJe8LG2pMu9EHs1WrUZErJNguu9Iu0+Qkg2x6TknsANY9KRXujdWyfqHiayMELJIQramV32LWaMc7GqCCCiIy46HAxsGe4imT7JkuizI6f0IjjwspnCjjCWe40Fh7hW4joSY9AWpMgAhMr4uaAR7L33k1o2ilFio42g5cGZKgiOdoajVpYmSLlH3fc6D1npcnDFmOMktuGtlCpjF1zYbPaQ3mcH2nYgUfVG24ikTrnpji6/tyHRCQoA7ovhcpAZoo+k88Ia9JK5y3qvA2ASBDBh7eyq7fUGnXweGZ2LxKEdAywOJX+tNB+qiTlXdijwIV2fZOUavJVoi2tCcrFEo9PWT+YyUw5BMsza93WLH82w8SL5N0v43AdTKstEMQMlyrlLcehjFl3KJILijSdtlSx+VHrRsP06xkpicJIxEk7keKHsWMLkcdI9lWYzilBvJ2grrfekpfXSOdmgw0g1hEk045/R2bBUim4wWR3kRDdptaqLpuia6lFYOWvrcg3WNcmcbcZ/NRULkj46rQ2tohnysmZdVaicm2tKnTT605rr0Rq9vgAyebDx74dOfsgE7dMgBTRFwjJ7J5ZzoFJLj6pzYBtIIrWuVZ4PJ5qNUTtNYci5Be1m0x1r2wN2bJzktWT2rMqeiUjQqV+XWByuqu5fzNsm8xse7EnKoS0XgKikea6IRUTzbrvd5qat/0ORxZ5uMKxshgKEGwiAZj2229Sb5kINWq5pIXbGG7y8tiFDUr1FFW+E6GRP7GtHOiAfmk/uSK+DB6fsYrrvCKKH5xnEenaGBa9Ck3aoF4xN8r7gJvDqREMN32JW9x9XdMkb"
)
MAX_LENGTH = 8192
GENERATION_SAMPLES = 2
GENERATION_TEMPERATURE = 0.6
GENERATION_TOP_P = 0.95
GENERATION_MAX_NEW_TOKENS = 1536
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 20260911

WORKING_ROOT = Path("/kaggle/working/aimo_e12_metacognitive_readout")
TEMP_ROOT = Path("/kaggle/temp/aimo_e12_metacognitive_readout")
REPOSITORY_DIR = TEMP_ROOT / "official-baselines"
MODEL_CACHE = TEMP_ROOT / "huggingface"
FEATURE_CACHE = WORKING_ROOT / "feature_cache"
GENERATION_CACHE = WORKING_ROOT / "generation_cache"
RESULT_DIR = WORKING_ROOT / "result"
FINAL_ZIP = Path("/kaggle/working/aimo_e12_metacognitive_readout_results.zip")
PREDICTION_FREEZE = RESULT_DIR / "external_predictions_frozen.csv"
PREDICTION_FREEZE_AUDIT = RESULT_DIR / "external_prediction_freeze.json"

OFFICIAL_SOLVE_PROMPT = (
    "You are a precise math problem solver. Solve the given problem step by step, "
    "then output your final answer on the last line in the exact format:\n"
    "ANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does not "
    "have a solution."
)
DIAGNOSTIC_SYSTEM_PROMPT = (
    "You are performing a behavioral forecast about this exact language model. "
    "Do not solve the mathematics. Follow the requested one-letter response format."
)
DIAGNOSTIC_PREFILL = "The more likely description is option"
RRB_SYSTEM_PROMPT = (
    "You are a helpful math assistant. Please reason step by step, and put your "
    "final answer within \\boxed{}.\n"
)
RRB_PROTOCOL_PREFIX = """YOUR PROTOCOL:
1. Read the "TRANSFORMATION RULE" provided by the user and reverse the transformation on the "TRANSFORMED INPUT" to obtain the reconstructed problem statement.
2. Once you have the reconstructed problem statement, proceed to solve the math problem.
3. Put your final answer within \\boxed{}.
"""
RRB_RULES = {
    "sentence_reversal": "The order of sentences in the user query has been reversed. Sentences are defined as sequences of symbols separated by periods.",
    "word_reversal": "The order of words (words are defined as sequences of symbols separated by spaces) in the user query has been reversed.",
    "split_reversal": "Every word (words are defined as sequences of symbols separated by spaces) in user query has its symbols in reverse order.",
}


def run(command: list[str], **kwargs) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_text(value: str) -> str:
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def normalized_text(value: str) -> str:
    return " ".join(canonical_text(value).split())


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def ensure_runtime_packages() -> None:
    required = {
        "transformers": "4.51",
        "accelerate": "0.30",
        "datasets": "3.0",
        "scikit-learn": "1.4",
    }
    missing = []
    for package, minimum in required.items():
        try:
            current = importlib.metadata.version(package)
            current_tuple = tuple(int(x) for x in re.findall(r"\d+", current)[:2])
            minimum_tuple = tuple(int(x) for x in minimum.split(".")[:2])
            if current_tuple < minimum_tuple:
                missing.append(f"{package}>={minimum}")
        except importlib.metadata.PackageNotFoundError:
            missing.append(f"{package}>={minimum}")
    if missing:
        run([sys.executable, "-m", "pip", "install", "--quiet", *missing])


def gpu_preflight() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device. Select Kaggle GPU T4 x2.")
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
    if len(devices) != 2 or any("T4" not in x["name"].upper() for x in devices):
        raise RuntimeError(f"E12 requires exactly Kaggle T4 x2; found {devices}")
    if any(x["memory_gib"] < 14.0 for x in devices):
        raise RuntimeError(f"Insufficient GPU memory: {devices}")
    result = {"torch": torch.__version__, "devices": devices}
    print(json.dumps(result, indent=2), flush=True)
    return result


def optional_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("HF_TOKEN").strip() or None
    except Exception:
        return None


def checkout_official() -> None:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    if not (REPOSITORY_DIR / ".git").exists():
        run(["git", "clone", "--filter=blob:none", OFFICIAL_REPOSITORY, str(REPOSITORY_DIR)])
    run(["git", "-C", str(REPOSITORY_DIR), "fetch", "--quiet", "origin"])
    run(["git", "-C", str(REPOSITORY_DIR), "checkout", "--quiet", "--detach", OFFICIAL_COMMIT])
    actual = subprocess.check_output(
        ["git", "-C", str(REPOSITORY_DIR), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != OFFICIAL_COMMIT:
        raise RuntimeError(f"official baseline commit mismatch: {actual}")
    solve_prompt = (REPOSITORY_DIR / "prompts" / "solve.txt").read_text(encoding="utf-8").strip()
    if solve_prompt != OFFICIAL_SOLVE_PROMPT:
        raise RuntimeError("pinned official solve prompt did not reproduce the frozen text")


def canonical_row_hash(rows: list[dict]) -> str:
    return sha256_bytes(stable_json(rows).encode("utf-8"))


def prepare_official_rows() -> tuple[list[dict], dict]:
    source = REPOSITORY_DIR / "data" / "math-robust-agg.csv"
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    required = {"problem_id", "dataset_id", "original_problem", "model_id", "model_is_robust"}
    if not raw or required.difference(raw[0]):
        raise RuntimeError("official aggregate dataset has unexpected schema")
    by_id: dict[str, dict] = {}
    models = set()
    for row in raw:
        problem_id = row["problem_id"].strip()
        item = {
            "problem_id": problem_id,
            "dataset_id": row["dataset_id"].strip(),
            "original_problem": canonical_text(row["original_problem"]),
            "model_is_robust": parse_bool(row["model_is_robust"]),
        }
        models.add(row["model_id"].strip())
        if problem_id in by_id:
            previous = by_id[problem_id]
            comparable = {**item, "original_problem": normalized_text(item["original_problem"])}
            previous_comparable = {
                **previous,
                "original_problem": normalized_text(previous["original_problem"]),
            }
            if comparable != previous_comparable:
                raise RuntimeError(f"inconsistent duplicate official problem {problem_id}")
        else:
            by_id[problem_id] = item
    rows = [by_id[key] for key in sorted(by_id)]
    digest = canonical_row_hash(rows)
    if len(rows) != 137 or digest != OFFICIAL_CANONICAL_SHA256:
        raise RuntimeError(f"official data drift: rows={len(rows)}, canonical_sha256={digest}")
    if models != {"qwen3-8b:low"}:
        raise RuntimeError(f"unexpected official target models: {sorted(models)}")
    audit = {
        "source_commit": OFFICIAL_COMMIT,
        "input_rows": len(raw),
        "unique_problem_rows": len(rows),
        "canonical_sha256": digest,
        "label_counts": dict(sorted(Counter(str(x["model_is_robust"]) for x in rows).items())),
        "source_counts": dict(sorted(Counter(x["dataset_id"] for x in rows).items())),
    }
    return rows, audit


def prepare_external_rows(official_rows: list[dict]) -> tuple[list[dict], dict]:
    from datasets import load_dataset

    dataset = load_dataset(AIME_DATASET_ID, revision=AIME_REVISION, split="train")
    required = {"id", "problem", "answer"}
    if len(dataset) != 30 or required.difference(dataset.column_names):
        raise RuntimeError(
            f"AIME 2024 drift: rows={len(dataset)}, columns={dataset.column_names}"
        )
    rows = []
    for raw in dataset:
        answer_text = str(raw["answer"]).strip()
        if not re.fullmatch(r"\d{3}", answer_text):
            raise RuntimeError(f"unexpected AIME answer format: {answer_text!r}")
        rows.append(
            {
                "problem_id": f"aime2024_{int(raw['id']):03d}",
                "dataset_id": "HuggingFaceH4/aime_2024",
                "original_problem": canonical_text(raw["problem"]),
                "gold_answer": answer_text,
            }
        )
    rows.sort(key=lambda x: x["problem_id"])
    if len({x["problem_id"] for x in rows}) != 30:
        raise RuntimeError("duplicate AIME 2024 problem ids")
    official_ids = {x["problem_id"] for x in official_rows}
    official_texts = {normalized_text(x["original_problem"]) for x in official_rows}
    id_overlap = sorted(official_ids.intersection(x["problem_id"] for x in rows))
    text_overlap = sorted(official_texts.intersection(normalized_text(x["original_problem"]) for x in rows))
    if id_overlap or text_overlap:
        raise RuntimeError(
            f"external leakage: id_overlap={id_overlap}, normalized_text_overlap={len(text_overlap)}"
        )
    audit_rows = [
        {
            "problem_id": x["problem_id"],
            "original_problem": x["original_problem"],
            "gold_answer": x["gold_answer"],
        }
        for x in rows
    ]
    audit = {
        "dataset_id": AIME_DATASET_ID,
        "revision": AIME_REVISION,
        "row_count": len(rows),
        "canonical_sha256": canonical_row_hash(audit_rows),
        "official_id_overlap": id_overlap,
        "official_normalized_text_overlap": len(text_overlap),
    }
    return rows, audit


def load_frozen_v6_oof_control(official_rows: list[dict]):
    """Load the exact V6 OOF predictions frozen before E11 existed."""
    import numpy as np

    raw = zlib.decompress(base64.b64decode(V6_OOF_CSV_ZLIB_BASE64))
    if sha256_bytes(raw) != V6_OOF_CSV_SHA256:
        raise RuntimeError("embedded V6 OOF control hash mismatch")
    records = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    by_id = {record["problem_id"]: record for record in records}
    if len(records) != 137 or len(by_id) != 137:
        raise RuntimeError("embedded V6 OOF control has unexpected cardinality")
    fractions = []
    labels = []
    for row in official_rows:
        record = by_id.get(row["problem_id"])
        if record is None:
            raise RuntimeError(f"V6 OOF control missing problem {row['problem_id']}")
        label = parse_bool(record["label"])
        if label != bool(row["model_is_robust"]):
            raise RuntimeError(f"V6 OOF control label mismatch for {row['problem_id']}")
        fractions.append(float(record["positive_vote_fraction"]))
        labels.append(label)
    predictions = np.asarray(fractions) >= 0.5
    y = np.asarray(labels, dtype=bool)
    score = balanced_accuracy(y, predictions)
    if not math.isclose(score, V6_REFERENCE_BA, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"frozen V6 OOF control failed: {score} != {V6_REFERENCE_BA}")
    return np.asarray(fractions, dtype=np.float64)


def load_deployed_v6_artifact() -> tuple[dict, dict]:
    """Download and validate the exact V6 artifact that scored 12/19."""
    import numpy as np

    archive_path = TEMP_ROOT / "aimo-small-v6-fixed-layer-vote-20260910.zip"
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(V6_PUBLIC_ZIP_URL, archive_path)
    archive_hash = sha256_file(archive_path)
    if archive_hash != V6_PUBLIC_ZIP_SHA256:
        raise RuntimeError(f"deployed V6 ZIP hash mismatch: {archive_hash}")
    with zipfile.ZipFile(archive_path) as archive:
        artifact_bytes = archive.read("probe_artifacts/probe_artifact.pkl")
    artifact_hash = sha256_bytes(artifact_bytes)
    if artifact_hash != V6_ARTIFACT_SHA256:
        raise RuntimeError(f"deployed V6 artifact hash mismatch: {artifact_hash}")
    # The pickle is loaded only after the immutable public byte hash is checked.
    artifact = pickle.loads(artifact_bytes)
    strategy = artifact.get("recommended_strategy", {})
    if (
        artifact.get("schema_version") != 3
        or artifact.get("artifact_type") != "fixed_multilayer_vote_probe_ensemble"
        or artifact.get("model_id") != MODEL_ID
        or artifact.get("system_prompt") != OFFICIAL_SOLVE_PROMPT
        or strategy.get("name") != "fixed_multilayer_majority_vote"
        or strategy.get("layers") != LAYERS
        or len(artifact.get("groups", [])) != 25
    ):
        raise RuntimeError("deployed V6 artifact schema or provenance mismatch")
    vote_count = 0
    for group in artifact["groups"]:
        if group.get("control_task") != "NONE":
            raise RuntimeError("deployed V6 artifact contains an ineligible control group")
        probes = group.get("probes", {})
        for layer in LAYERS:
            probe = probes.get(str(layer), probes.get(layer))
            if probe is None:
                raise RuntimeError(f"deployed V6 artifact is missing layer {layer}")
            weights = np.asarray(probe["weights"], dtype=np.float32)
            if weights.ndim == 1:
                weights = weights.reshape(1, -1)
            if weights.ndim != 2 or weights.shape[1] != 4096:
                raise RuntimeError(f"invalid V6 weights at layer {layer}: {weights.shape}")
            vote_count += weights.shape[0]
    if vote_count != 225:
        raise RuntimeError(f"deployed V6 vote count mismatch: {vote_count}")
    audit = {
        "repository_commit": V6_PUBLIC_REPOSITORY_COMMIT,
        "url": V6_PUBLIC_ZIP_URL,
        "zip_sha256": archive_hash,
        "artifact_sha256": artifact_hash,
        "group_count": len(artifact["groups"]),
        "vote_count": vote_count,
        "codabench_score": "12/19",
    }
    return artifact, audit


def deployed_v6_vote_fraction(original_cube, artifact: dict):
    """Apply the real submission's 225 hard votes to a batch of representations."""
    import numpy as np

    votes = []
    for group in artifact["groups"]:
        probes = group["probes"]
        for layer_position, layer in enumerate(LAYERS):
            probe = probes.get(str(layer), probes.get(layer))
            weights = np.asarray(probe["weights"], dtype=np.float32)
            if weights.ndim == 1:
                weights = weights.reshape(1, -1)
            bias = np.asarray(probe["bias"], dtype=np.float32).reshape(-1)
            threshold = np.asarray(probe["threshold"], dtype=np.float32).reshape(-1)
            if bias.size != weights.shape[0] or threshold.size != weights.shape[0]:
                raise RuntimeError("deployed V6 probe arrays have inconsistent shapes")
            margins = original_cube[:, layer_position, :].astype(np.float32) @ weights.T
            margins += bias.reshape(1, -1) - threshold.reshape(1, -1)
            votes.append(margins >= 0.0)
    matrix = np.concatenate(votes, axis=1)
    if matrix.shape != (original_cube.shape[0], 225):
        raise RuntimeError(f"unexpected deployed V6 vote matrix: {matrix.shape}")
    return matrix.mean(axis=1, dtype=np.float64)


def diagnostic_prompt(problem: str, view: str) -> str:
    """Return one of four fixed, counterbalanced behavioral forecasts."""
    problem = canonical_text(problem)
    if view.startswith("robustness_"):
        positive = (
            "The model would give the same final answer across several "
            "meaning-preserving paraphrases of the problem."
        )
        negative = (
            "The model would give different final answers for at least one "
            "meaning-preserving paraphrase of the problem."
        )
        question = (
            "Predict paraphrase stability for this exact language model. "
            "Do not solve or rewrite the problem."
        )
        positive_is_a = view == "robustness_stable_a"
    elif view.startswith("correctness_"):
        positive = "The model would give the mathematically correct final answer on its first attempt."
        negative = "The model would give an incorrect final answer on its first attempt."
        question = (
            "Predict first-attempt correctness for this exact language model. "
            "Do not solve or rewrite the problem."
        )
        positive_is_a = view == "correctness_correct_a"
    else:
        raise ValueError(f"unknown diagnostic view: {view}")
    option_a, option_b = (positive, negative) if positive_is_a else (negative, positive)
    return (
        f"MATHEMATICS PROBLEM:\n{problem}\n\n"
        f"DIAGNOSTIC TASK:\n{question}\n"
        f"A. {option_a}\n"
        f"B. {option_b}\n"
        "Return only A or B."
    )


def remove_latex_comments(text: str) -> str:
    def clean_line(line: str) -> str:
        def replacer(match: re.Match) -> str:
            return match.group(1) if match.group(1) else ""

        return re.sub(r"(\\\\|\\%)|(%.*)", replacer, line)

    return "\n".join(clean_line(line) for line in text.split("\n"))


def rrb_clean(problem: str) -> str:
    text = remove_latex_comments(canonical_text(problem))
    text = re.sub(r"([bntafr])\\", r"\1 \\", text)
    return text.replace("\n", "; ")


def sentence_reversal(text: str) -> str:
    parts = text.split(".")[::-1]
    if parts and parts[0] == "":
        parts = parts[1:] + [""]
    return ".".join(parts)


def word_reversal(text: str) -> str:
    fragments = re.split(r"(\s+)", text)
    words = [x for x in fragments if x and not x.isspace()][::-1]
    cursor = iter(words)
    return "".join(x if x.isspace() else next(cursor) for x in fragments if x)


def split_reversal(text: str) -> str:
    return " ".join(part[::-1] for part in text.split(" "))


def rrb_transformed(problem: str, variant: str) -> str:
    clean = rrb_clean(problem)
    if variant == "baseline":
        return clean.strip()
    functions = {
        "sentence_reversal": sentence_reversal,
        "word_reversal": word_reversal,
        "split_reversal": split_reversal,
    }
    transformed = functions[variant](clean)
    return (
        RRB_PROTOCOL_PREFIX
        + "\nTRANSFORMATION RULE:\n"
        + RRB_RULES[variant]
        + "\n\nTRANSFORMED INPUT:\n"
        + transformed
    ).strip()


def validate_transformations(rows: list[dict]) -> dict:
    failures = []
    functions = {
        "sentence_reversal": sentence_reversal,
        "word_reversal": word_reversal,
        "split_reversal": split_reversal,
    }
    for row in rows:
        clean = rrb_clean(row["original_problem"])
        for name, function in functions.items():
            recovered = function(function(clean))
            if recovered != clean:
                failures.append({"problem_id": row["problem_id"], "variant": name})
    if failures:
        raise RuntimeError(f"RRB reversibility audit failed: {failures[:5]}")
    return {
        "repository": RRB_REPOSITORY,
        "commit": RRB_COMMIT,
        "variants": list(functions),
        "tested_problems": len(rows),
        "self_inverse_checks": len(rows) * len(functions),
        "passed": True,
    }


def build_chat_prompt(tokenizer, user_text: str, system_prompt: str) -> str:
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


def model_devices(model) -> dict:
    mapping = getattr(model, "hf_device_map", None)
    if not isinstance(mapping, dict):
        raise RuntimeError("Transformers did not create a multi-GPU device map")
    placements = {str(value) for value in mapping.values()}
    if "cpu" in placements or "disk" in placements:
        raise RuntimeError(f"CPU/disk offload refused: {sorted(placements)}")
    gpu_ids = set()
    for value in mapping.values():
        match = re.search(r"(?:cuda:)?(\d+)$", str(value))
        if match:
            gpu_ids.add(int(match.group(1)))
    if gpu_ids != {0, 1}:
        raise RuntimeError(f"both T4 GPUs must be used; map={mapping}")
    return {str(key): str(value) for key, value in mapping.items()}


def feature_fingerprint(row: dict, view: str) -> str:
    payload = {
        "schema": "e12_counterbalanced_logit_lens_v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "system_prompt": DIAGNOSTIC_SYSTEM_PROMPT,
        "assistant_prefill": DIAGNOSTIC_PREFILL,
        "problem_id": row["problem_id"],
        "view": view,
        "view_text": diagnostic_prompt(row["original_problem"], view),
        "layers": LAYERS,
        "max_length": MAX_LENGTH,
        "dtype": "float16",
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def feature_cache_path(split: str, row: dict, view: str, digest: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", row["problem_id"])
    return FEATURE_CACHE / split / f"{safe}-{view}-{digest[:16]}.npz"


def feature_cache_valid(path: Path, digest: str) -> bool:
    import numpy as np

    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                str(data["fingerprint"].item()) == digest
                and data["semantic_margins"].shape == (len(LAYERS),)
                and np.isfinite(data["semantic_margins"]).all()
            )
    except Exception:
        return False


def load_exact_model(token: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    common = {
        "revision": MODEL_REVISION,
        "cache_dir": MODEL_CACHE,
        "token": token,
    }
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **common)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="balanced",
        max_memory={0: "13GiB", 1: "13GiB", "cpu": "20GiB"},
        low_cpu_mem_usage=True,
        **common,
    )
    placements = model_devices(model)
    model.eval()
    return tokenizer, model, placements


def option_token_ids(tokenizer) -> tuple[int, int]:
    encoded = {
        letter: tokenizer.encode(" " + letter, add_special_tokens=False)
        for letter in ("A", "B")
    }
    if any(len(tokens) != 1 for tokens in encoded.values()):
        raise RuntimeError(f"A/B options are not single tokens: {encoded}")
    if encoded["A"][0] == encoded["B"][0]:
        raise RuntimeError("A/B option token ids unexpectedly collide")
    return int(encoded["A"][0]), int(encoded["B"][0])


def final_norm_module(model):
    core = getattr(model, "model", None)
    norm = getattr(core, "norm", None)
    if norm is None:
        nested = getattr(core, "model", None)
        norm = getattr(nested, "norm", None)
    if norm is None:
        raise RuntimeError("unable to locate the target model final normalization layer")
    return norm


def semantic_layer_margins(model, hidden_states, token_a: int, token_b: int, positive_is_a: bool):
    import numpy as np
    import torch

    norm = final_norm_module(model)
    output_head = model.get_output_embeddings()
    norm_device = next(norm.parameters()).device
    # Accelerate can leave requested layer states on different GPUs. Move each
    # vector separately before stacking rather than attempting a cross-device stack.
    vectors = torch.stack(
        [hidden_states[index][0, -1, :].to(norm_device) for index in LAYERS]
    )
    normalized = norm(vectors)
    head_device = output_head.weight.device
    normalized = normalized.to(head_device)
    option_ids = torch.tensor([token_a, token_b], device=head_device, dtype=torch.long)
    option_weights = output_head.weight.index_select(0, option_ids)
    logits = normalized.float() @ option_weights.float().T
    bias = getattr(output_head, "bias", None)
    if bias is not None:
        logits = logits + bias.index_select(0, option_ids).float()
    raw = logits[:, 0] - logits[:, 1]
    semantic = raw if positive_is_a else -raw
    result = semantic.detach().cpu().numpy().astype(np.float32)
    if result.shape != (len(LAYERS),) or not np.isfinite(result).all():
        raise RuntimeError(f"invalid semantic logit margins: {result.shape}")
    return result


def extract_diagnostic_views(tokenizer, model, split: str, rows: list[dict]) -> dict:
    import numpy as np
    import torch

    FEATURE_CACHE.mkdir(parents=True, exist_ok=True)
    input_device = model.get_input_embeddings().weight.device
    token_a, token_b = option_token_ids(tokenizer)
    jobs = []
    for row in rows:
        for view in DIAGNOSTIC_VIEWS:
            digest = feature_fingerprint(row, view)
            path = feature_cache_path(split, row, view, digest)
            if not feature_cache_valid(path, digest):
                jobs.append((row, view, digest, path))
    total = len(rows) * len(DIAGNOSTIC_VIEWS)
    print(f"Feature extraction: {total - len(jobs)}/{total} cached; {len(jobs)} pending", flush=True)
    started = time.time()
    for number, (row, view, digest, path) in enumerate(jobs, start=1):
        user_text = diagnostic_prompt(row["original_problem"], view)
        prompt = build_chat_prompt(tokenizer, user_text, DIAGNOSTIC_SYSTEM_PROMPT)
        prompt += DIAGNOSTIC_PREFILL
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        token_count = int(encoded["input_ids"].shape[1])
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
        if output.hidden_states is None or len(output.hidden_states) != 37:
            raise RuntimeError("expected embedding output plus 36 layer states")
        positive_is_a = view.endswith("_a")
        margins = semantic_layer_margins(
            model, output.hidden_states, token_a, token_b, positive_is_a
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                semantic_margins=margins,
                token_count=np.asarray(token_count),
                fingerprint=np.asarray(digest),
            )
        temporary.replace(path)
        del output, encoded, margins
        if number == 1 or number % 10 == 0 or number == len(jobs):
            rate = (time.time() - started) / number
            print(
                f"[{number}/{len(jobs)}] {split}/{row['problem_id']}/{view} "
                f"tokens={token_count}, mean={rate:.2f}s/job",
                flush=True,
            )
    return {
        "split": split,
        "total_jobs": total,
        "new_jobs": len(jobs),
        "cached_jobs": total - len(jobs),
        "option_token_ids": {"A": token_a, "B": token_b},
    }


def load_diagnostic_cube(split: str, rows: list[dict]):
    import numpy as np

    matrices = []
    token_counts = []
    for row in rows:
        per_view = []
        per_counts = []
        for view in DIAGNOSTIC_VIEWS:
            digest = feature_fingerprint(row, view)
            path = feature_cache_path(split, row, view, digest)
            if not feature_cache_valid(path, digest):
                raise RuntimeError(f"missing feature cache: {split}/{row['problem_id']}/{view}")
            with np.load(path, allow_pickle=False) as data:
                per_view.append(data["semantic_margins"].astype(np.float32))
                per_counts.append(int(data["token_count"].item()))
        matrices.append(np.stack(per_view))
        token_counts.append(per_counts)
    cube = np.stack(matrices)
    if cube.shape != (len(rows), len(DIAGNOSTIC_VIEWS), len(LAYERS)):
        raise RuntimeError(f"unexpected diagnostic cube shape: {cube.shape}")
    return cube, token_counts


def make_metacognitive_features(cube):
    import numpy as np

    robustness = cube[:, 0:2, :].mean(axis=1, dtype=np.float64)
    correctness = cube[:, 2:4, :].mean(axis=1, dtype=np.float64)
    contrast = robustness - correctness
    matrix = np.concatenate([robustness, contrast], axis=1).astype(np.float32)
    names = [f"robustness_semantic_margin_layer_{layer}" for layer in LAYERS]
    names += [f"robustness_minus_correctness_layer_{layer}" for layer in LAYERS]
    if matrix.shape != (cube.shape[0], 18) or not np.isfinite(matrix).all():
        raise RuntimeError(f"invalid E12 feature matrix: {matrix.shape}")
    return matrix, names


def option_order_diagnostics(cube) -> dict:
    import numpy as np

    robust_a, robust_b = cube[:, 0, :], cube[:, 1, :]
    correct_a, correct_b = cube[:, 2, :], cube[:, 3, :]

    def summarize(left, right) -> dict:
        flat_left = left.reshape(-1)
        flat_right = right.reshape(-1)
        if np.std(flat_left) < 1e-12 or np.std(flat_right) < 1e-12:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(flat_left, flat_right)[0, 1])
        return {
            "mean_absolute_margin_difference": float(np.mean(np.abs(left - right))),
            "semantic_sign_agreement": float(np.mean((left >= 0) == (right >= 0))),
            "pearson_correlation": correlation,
        }

    return {
        "robustness": summarize(robust_a, robust_b),
        "correctness": summarize(correct_a, correct_b),
    }


def original_state_fingerprint(row: dict) -> str:
    payload = {
        "schema": "e12_external_original_states_v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "system_prompt": OFFICIAL_SOLVE_PROMPT,
        "problem_id": row["problem_id"],
        "problem": canonical_text(row["original_problem"]),
        "layers": LAYERS,
        "max_length": MAX_LENGTH,
        "dtype": "float16",
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def original_state_cache_path(row: dict, digest: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", row["problem_id"])
    return FEATURE_CACHE / "external_v6_original" / f"{safe}-{digest[:16]}.npz"


def original_state_cache_valid(path: Path, digest: str) -> bool:
    import numpy as np

    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                str(data["fingerprint"].item()) == digest
                and data["hidden_states"].shape == (len(LAYERS), 4096)
                and np.isfinite(data["hidden_states"]).all()
            )
    except Exception:
        return False


def extract_external_original_states(tokenizer, model, rows: list[dict]) -> dict:
    """Run only after Stage A passes; needed solely for deployed-V6 comparison."""
    import numpy as np
    import torch

    input_device = model.get_input_embeddings().weight.device
    jobs = []
    for row in rows:
        digest = original_state_fingerprint(row)
        path = original_state_cache_path(row, digest)
        if not original_state_cache_valid(path, digest):
            jobs.append((row, digest, path))
    print(
        f"External V6 original states: {len(rows) - len(jobs)}/{len(rows)} cached; "
        f"{len(jobs)} pending",
        flush=True,
    )
    started = time.time()
    for number, (row, digest, path) in enumerate(jobs, start=1):
        prompt = build_chat_prompt(
            tokenizer, canonical_text(row["original_problem"]), OFFICIAL_SOLVE_PROMPT
        )
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        token_count = int(encoded["input_ids"].shape[1])
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
        if output.hidden_states is None or len(output.hidden_states) != 37:
            raise RuntimeError("expected embedding output plus 36 layer states")
        matrix = np.stack(
            [output.hidden_states[index][0, -1, :].detach().float().cpu().numpy() for index in LAYERS]
        ).astype(np.float16)
        if matrix.shape != (len(LAYERS), 4096) or not np.isfinite(matrix).all():
            raise RuntimeError(f"invalid external original states for {row['problem_id']}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                hidden_states=matrix,
                token_count=np.asarray(token_count),
                fingerprint=np.asarray(digest),
            )
        temporary.replace(path)
        del output, encoded, matrix
        if number == 1 or number % 10 == 0 or number == len(jobs):
            rate = (time.time() - started) / number
            print(f"[{number}/{len(jobs)}] external-v6/{row['problem_id']}; mean={rate:.2f}s/job", flush=True)
    return {"total_jobs": len(rows), "new_jobs": len(jobs), "cached_jobs": len(rows) - len(jobs)}


def load_external_original_cube(rows: list[dict]):
    import numpy as np

    matrices = []
    token_counts = []
    for row in rows:
        digest = original_state_fingerprint(row)
        path = original_state_cache_path(row, digest)
        if not original_state_cache_valid(path, digest):
            raise RuntimeError(f"missing external original state cache for {row['problem_id']}")
        with np.load(path, allow_pickle=False) as data:
            matrices.append(data["hidden_states"].astype(np.float32))
            token_counts.append(int(data["token_count"].item()))
    cube = np.stack(matrices)
    if cube.shape != (len(rows), len(LAYERS), 4096):
        raise RuntimeError(f"unexpected external original cube shape: {cube.shape}")
    return cube, token_counts


def split_indices(y, groups, seed: int):
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold

    indices = np.arange(len(y))
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    dummy = np.zeros((len(y), 1), dtype=np.float32)
    for train, test in splitter.split(dummy, y, groups):
        if set(groups[train]).intersection(groups[test]):
            raise RuntimeError("problem-group leakage")
        yield train, test


def fit_candidate(x_train, y_train, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(
        C=CANDIDATE_C,
        class_weight="balanced",
        penalty="l2",
        solver="liblinear",
        max_iter=5000,
        random_state=seed,
    )
    model.fit(scaler.transform(x_train), y_train)
    return scaler, model


def candidate_oof(x, y, groups, external_x=None, seeds=SEEDS):
    import numpy as np

    per_row = [[] for _ in range(len(y))]
    fold_records = []
    for seed in seeds:
        for fold, (train, test) in enumerate(split_indices(y, groups, seed)):
            scaler, model = fit_candidate(x[train], y[train], seed)
            probabilities = model.predict_proba(scaler.transform(x[test]))[:, 1]
            for index, value in zip(test, probabilities, strict=True):
                per_row[int(index)].append(float(value))
            fold_records.append(
                {"seed": seed, "fold": fold, "train_count": len(train), "test_count": len(test)}
            )
    if any(len(values) != len(seeds) for values in per_row):
        raise RuntimeError("candidate OOF coverage failure")
    oof = np.asarray([np.mean(values) for values in per_row])
    external = None
    if external_x is not None:
        external_probabilities = []
        for seed in seeds:
            scaler, model = fit_candidate(x, y, seed)
            external_probabilities.append(model.predict_proba(scaler.transform(external_x))[:, 1])
        external = np.mean(np.stack(external_probabilities), axis=0)
    return oof, external, fold_records


def fit_v6_predictions(x_train, y_train, x_test, seed: int):
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    mean = x_train.mean(axis=0, dtype=np.float64)
    scale = x_train.std(axis=0, dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    train_z = ((x_train - mean) / scale).astype(np.float32)
    test_z = ((x_test - mean) / scale).astype(np.float32)
    model = LogisticRegression(
        C=V6_C,
        class_weight="balanced",
        dual=True,
        max_iter=5000,
        penalty="l2",
        random_state=seed,
        solver="liblinear",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
        model.fit(train_z, y_train)
    return model.decision_function(test_z) >= 0.0


def v6_oof(original_cube, y, groups, external_original=None):
    import numpy as np

    per_row = [[] for _ in range(len(y))]
    for seed in SEEDS:
        for train, test in split_indices(y, groups, seed):
            layer_votes = []
            for layer_position in range(len(LAYERS)):
                layer_votes.append(
                    fit_v6_predictions(
                        original_cube[train, layer_position, :],
                        y[train],
                        original_cube[test, layer_position, :],
                        seed,
                    ).astype(np.float64)
                )
            fold_fraction = np.mean(np.stack(layer_votes), axis=0)
            for index, value in zip(test, fold_fraction, strict=True):
                per_row[int(index)].append(float(value))
    if any(len(values) != len(SEEDS) for values in per_row):
        raise RuntimeError("V6 OOF coverage failure")
    oof = np.asarray([np.mean(values) for values in per_row])
    external = None
    if external_original is not None:
        votes = []
        for seed in SEEDS:
            for layer_position in range(len(LAYERS)):
                votes.append(
                    fit_v6_predictions(
                        original_cube[:, layer_position, :],
                        y,
                        external_original[:, layer_position, :],
                        seed,
                    ).astype(np.float64)
                )
        external = np.mean(np.stack(votes), axis=0)
    return oof, external


def balanced_accuracy(y, predictions) -> float:
    import numpy as np

    truth = np.asarray(y, dtype=bool)
    predicted = np.asarray(predictions, dtype=bool)
    if truth.shape != predicted.shape or truth.size == 0:
        raise ValueError("balanced accuracy inputs must have the same non-empty shape")
    positives = truth
    negatives = ~truth
    if not positives.any() or not negatives.any():
        raise ValueError("balanced accuracy requires both classes")
    true_positive_rate = float(np.mean(predicted[positives]))
    true_negative_rate = float(np.mean(~predicted[negatives]))
    return (true_positive_rate + true_negative_rate) / 2.0


def bootstrap_score_lower(y, predictions, seed=BOOTSTRAP_SEED) -> float:
    import numpy as np

    generator = np.random.default_rng(seed)
    scores = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = generator.integers(0, len(y), len(y))
        if len(np.unique(y[sample])) == 2:
            scores.append(balanced_accuracy(y[sample], predictions[sample]))
    return float(np.quantile(scores, 0.025))


def paired_bootstrap_lower(y, candidate, control) -> float:
    import numpy as np

    generator = np.random.default_rng(BOOTSTRAP_SEED + 1)
    differences = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = generator.integers(0, len(y), len(y))
        if len(np.unique(y[sample])) == 2:
            differences.append(
                balanced_accuracy(y[sample], candidate[sample])
                - balanced_accuracy(y[sample], control[sample])
            )
    return float(np.quantile(differences, 0.025))


def source_holdout_scores(x, y, sources) -> dict[str, float]:
    import numpy as np

    unique = sorted(set(sources.tolist()))
    if len(unique) != 2:
        raise RuntimeError(f"expected two official sources, found {unique}")
    result = {}
    for train_source, test_source in ((unique[0], unique[1]), (unique[1], unique[0])):
        train = np.flatnonzero(sources == train_source)
        test = np.flatnonzero(sources == test_source)
        probabilities = []
        for seed in SEEDS:
            scaler, model = fit_candidate(x[train], y[train], seed)
            probabilities.append(model.predict_proba(scaler.transform(x[test]))[:, 1])
        prediction = np.mean(np.stack(probabilities), axis=0) >= 0.5
        result[f"{train_source} -> {test_source}"] = balanced_accuracy(y[test], prediction)
    return result


def stage_a_validation(
    official_rows,
    official_cube,
    feature_names,
):
    import numpy as np
    from sklearn.metrics import confusion_matrix

    y = np.asarray([bool(x["model_is_robust"]) for x in official_rows])
    groups = np.asarray([x["problem_id"] for x in official_rows])
    sources = np.asarray([x["dataset_id"] for x in official_rows])
    official_x, names = make_metacognitive_features(official_cube)
    if names != feature_names:
        raise RuntimeError("feature-name mismatch")

    candidate_probability, _, folds = candidate_oof(official_x, y, groups)
    candidate_prediction = candidate_probability >= 0.5
    v6_fraction = load_frozen_v6_oof_control(official_rows)
    v6_prediction = v6_fraction >= 0.5
    v6_score = balanced_accuracy(y, v6_prediction)

    candidate_score = balanced_accuracy(y, candidate_prediction)
    lower = bootstrap_score_lower(y, candidate_prediction)
    paired_lower = paired_bootstrap_lower(y, candidate_prediction, v6_prediction)
    holdouts = source_holdout_scores(official_x, y, sources)
    holdout_mean = float(np.mean(list(holdouts.values())))
    randomized_y = np.random.default_rng(BOOTSTRAP_SEED).permutation(y)
    randomized_probability, _, _ = candidate_oof(
        official_x, randomized_y, groups, external_x=None, seeds=[SEEDS[0]]
    )
    randomized_score = balanced_accuracy(randomized_y, randomized_probability >= 0.5)
    control_advantage = candidate_score - randomized_score
    disagreement = float(np.mean(candidate_prediction != v6_prediction))
    tn, fp, fn, tp = confusion_matrix(y, candidate_prediction).ravel()

    failures = []
    minimum_score = V6_REFERENCE_BA + 0.02
    if candidate_score < minimum_score:
        failures.append(f"candidate BA {candidate_score:.6f} < {minimum_score:.6f}")
    if lower <= V6_REFERENCE_LOWER:
        failures.append(f"candidate bootstrap lower {lower:.6f} <= {V6_REFERENCE_LOWER:.6f}")
    if paired_lower <= 0.0:
        failures.append(f"paired bootstrap improvement lower {paired_lower:.6f} <= 0")
    for direction, score in holdouts.items():
        if score < 0.65:
            failures.append(f"source holdout {direction} {score:.6f} < 0.65")
    if holdout_mean < 0.69:
        failures.append(f"mean source holdout {holdout_mean:.6f} < 0.69")
    if control_advantage < 0.10:
        failures.append(f"random-control advantage {control_advantage:.6f} < 0.10")
    if disagreement < 0.08:
        failures.append(f"OOF disagreement vs V6 {disagreement:.6f} < 0.08")

    result = {
        "stage": "A",
        "passed": not failures,
        "failures": failures,
        "feature_names": feature_names,
        "fold_records": folds,
        "metrics": {
            "candidate_balanced_accuracy": candidate_score,
            "candidate_ordinary_accuracy": float(np.mean(candidate_prediction == y)),
            "candidate_bootstrap_95_lower": lower,
            "v6_frozen_oof_balanced_accuracy": v6_score,
            "paired_improvement_bootstrap_95_lower": paired_lower,
            "source_holdout_balanced_accuracy": holdouts,
            "mean_source_holdout_balanced_accuracy": holdout_mean,
            "randomized_control_balanced_accuracy": randomized_score,
            "randomized_control_advantage": control_advantage,
            "disagreement_vs_v6": disagreement,
            "predicted_positive": int(candidate_prediction.sum()),
            "actual_positive": int(y.sum()),
            "confusion_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
            "option_order_diagnostics": option_order_diagnostics(official_cube),
        },
        "thresholds": {
            "candidate_balanced_accuracy_min": minimum_score,
            "candidate_bootstrap_lower_strictly_above": V6_REFERENCE_LOWER,
            "paired_improvement_lower_strictly_above": 0.0,
            "source_holdout_each_min": 0.65,
            "source_holdout_mean_min": 0.69,
            "random_control_advantage_min": 0.10,
            "disagreement_vs_v6_min": 0.08,
        },
    }
    return result


def build_external_predictions(
    official_rows,
    official_cube,
    external_rows,
    external_cube,
    external_original_cube,
    deployed_v6_artifact,
) -> list[dict]:
    import numpy as np

    y = np.asarray([bool(x["model_is_robust"]) for x in official_rows])
    groups = np.asarray([x["problem_id"] for x in official_rows])
    official_x, names = make_metacognitive_features(official_cube)
    external_x, external_names = make_metacognitive_features(external_cube)
    if names != external_names:
        raise RuntimeError("external feature-name mismatch")
    _, external_probability, _ = candidate_oof(official_x, y, groups, external_x)
    if external_probability is None:
        raise RuntimeError("candidate external predictions were not produced")
    external_v6_fraction = deployed_v6_vote_fraction(
        external_original_cube, deployed_v6_artifact
    )
    predictions = []
    for row, candidate_p, v6_p in zip(
        external_rows, external_probability, external_v6_fraction, strict=True
    ):
        predictions.append(
            {
                "problem_id": row["problem_id"],
                "candidate_probability": float(candidate_p),
                "candidate_prediction": bool(candidate_p >= 0.5),
                "v6_vote_fraction": float(v6_p),
                "v6_prediction": bool(v6_p >= 0.5),
            }
        )
    return predictions


def write_prediction_freeze(predictions: list[dict]) -> dict:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "problem_id",
        "candidate_probability",
        "candidate_prediction",
        "v6_vote_fraction",
        "v6_prediction",
    ]
    rendered_rows = []
    for row in predictions:
        rendered_rows.append(
            {
                "problem_id": row["problem_id"],
                "candidate_probability": f"{row['candidate_probability']:.17g}",
                "candidate_prediction": str(row["candidate_prediction"]).lower(),
                "v6_vote_fraction": f"{row['v6_vote_fraction']:.17g}",
                "v6_prediction": str(row["v6_prediction"]).lower(),
            }
        )
    if PREDICTION_FREEZE.exists():
        existing_hash = sha256_file(PREDICTION_FREEZE)
        temporary = PREDICTION_FREEZE.with_suffix(".candidate.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rendered_rows)
        candidate_hash = sha256_file(temporary)
        temporary.unlink()
        if candidate_hash != existing_hash:
            raise RuntimeError("frozen external predictions changed on resume")
    else:
        with PREDICTION_FREEZE.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rendered_rows)
    audit = {
        "created_before_external_robustness_generation": not any(
            GENERATION_CACHE.glob("*.json")
        ),
        "row_count": len(predictions),
        "sha256": sha256_file(PREDICTION_FREEZE),
        "path": str(PREDICTION_FREEZE),
    }
    if (
        not audit["created_before_external_robustness_generation"]
        and not PREDICTION_FREEZE_AUDIT.exists()
    ):
        raise RuntimeError("generation cache predates the external prediction freeze")
    if PREDICTION_FREEZE_AUDIT.exists():
        previous = json.loads(PREDICTION_FREEZE_AUDIT.read_text(encoding="utf-8"))
        if previous["sha256"] != audit["sha256"]:
            raise RuntimeError("external prediction freeze audit hash changed")
        audit = previous
    else:
        PREDICTION_FREEZE_AUDIT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def extract_boxed(text: str) -> str | None:
    marker = "\\boxed{"
    starts = [match.start() for match in re.finditer(re.escape(marker), text)]
    for start in reversed(starts):
        cursor = start + len(marker)
        depth = 1
        for index in range(cursor, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[cursor:index]
    return None


def extract_aime_answer(text: str) -> int | None:
    boxed = extract_boxed(text)
    candidates = []
    if boxed is not None:
        compact = boxed.replace(" ", "").replace("$", "").replace(",", "")
        direct = re.fullmatch(r"([+-]?\d{1,4})", compact)
        fraction = re.fullmatch(r"\\(?:d?frac)\{([+-]?\d+)\}\{([+-]?\d+)\}", compact)
        if direct:
            candidates.append(direct.group(1))
        elif fraction and int(fraction.group(2)) != 0:
            numerator = int(fraction.group(1))
            denominator = int(fraction.group(2))
            if numerator % denominator == 0:
                candidates.append(str(numerator // denominator))
        else:
            candidates.extend(re.findall(r"(?<!\d)\d{1,3}(?!\d)", compact))
    if not candidates:
        answer_lines = re.findall(
            r"(?:final\s+answer|answer)\s*(?:is|:|=)?\s*\$?(-?\d{1,4})",
            text,
            flags=re.IGNORECASE,
        )
        candidates.extend(answer_lines[-1:])
    if not candidates:
        candidates.extend(re.findall(r"(?<!\d)-?\d{1,4}(?!\d)", text)[-1:])
    if not candidates:
        return None
    value = int(candidates[-1])
    return value if 0 <= value <= 999 else None


def generation_fingerprint(row: dict, variant: str) -> str:
    payload = {
        "schema": "e12_rrb_generation_v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "rrb_commit": RRB_COMMIT,
        "problem_id": row["problem_id"],
        "variant": variant,
        "system_prompt": RRB_SYSTEM_PROMPT,
        "user_prompt": rrb_transformed(row["original_problem"], variant),
        "samples": GENERATION_SAMPLES,
        "temperature": GENERATION_TEMPERATURE,
        "top_p": GENERATION_TOP_P,
        "max_new_tokens": GENERATION_MAX_NEW_TOKENS,
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def generation_cache_path(row: dict, variant: str, digest: str) -> Path:
    return GENERATION_CACHE / f"{row['problem_id']}-{variant}-{digest[:16]}.json"


def generation_cache_valid(path: Path, digest: str) -> bool:
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value["fingerprint"] == digest and len(value["responses"]) == GENERATION_SAMPLES
    except Exception:
        return False


def generate_external_labels(tokenizer, model, rows: list[dict]) -> list[dict]:
    import torch

    GENERATION_CACHE.mkdir(parents=True, exist_ok=True)
    input_device = model.get_input_embeddings().weight.device
    jobs = []
    for row in rows:
        for variant in RRB_VARIANTS:
            digest = generation_fingerprint(row, variant)
            path = generation_cache_path(row, variant, digest)
            if not generation_cache_valid(path, digest):
                jobs.append((row, variant, digest, path))
    print(
        f"External generation: {len(rows) * len(RRB_VARIANTS) - len(jobs)}/"
        f"{len(rows) * len(RRB_VARIANTS)} prompt groups cached; {len(jobs)} pending",
        flush=True,
    )
    started = time.time()
    for number, (row, variant, digest, path) in enumerate(jobs, start=1):
        user_prompt = rrb_transformed(row["original_problem"], variant)
        prompt = build_chat_prompt(tokenizer, user_prompt, RRB_SYSTEM_PROMPT)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        input_length = int(encoded["input_ids"].shape[1])
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        seed_material = f"{BOOTSTRAP_SEED}:{row['problem_id']}:{variant}".encode("utf-8")
        generation_seed = int(hashlib.sha256(seed_material).hexdigest()[:8], 16)
        torch.manual_seed(generation_seed)
        torch.cuda.manual_seed_all(generation_seed)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=GENERATION_TEMPERATURE,
                top_p=GENERATION_TOP_P,
                max_new_tokens=GENERATION_MAX_NEW_TOKENS,
                num_return_sequences=GENERATION_SAMPLES,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        responses = [
            tokenizer.decode(sequence[input_length:], skip_special_tokens=True)
            for sequence in generated
        ]
        answers = [extract_aime_answer(response) for response in responses]
        record = {
            "fingerprint": digest,
            "problem_id": row["problem_id"],
            "variant": variant,
            "generation_seed": generation_seed,
            "input_tokens": input_length,
            "responses": responses,
            "extracted_answers": answers,
        }
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        del generated, encoded
        if number == 1 or number % 5 == 0 or number == len(jobs):
            rate = (time.time() - started) / number
            eta = rate * (len(jobs) - number)
            print(
                f"[{number}/{len(jobs)}] {row['problem_id']}/{variant}; "
                f"answers={answers}; mean={rate:.1f}s/group; eta={eta/3600:.2f}h",
                flush=True,
            )
    records = []
    for row in rows:
        for variant in RRB_VARIANTS:
            digest = generation_fingerprint(row, variant)
            path = generation_cache_path(row, variant, digest)
            if not generation_cache_valid(path, digest):
                raise RuntimeError(f"missing generation cache for {row['problem_id']}/{variant}")
            records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def stage_b_validation(rows: list[dict], predictions: list[dict], generation_records: list[dict]):
    import numpy as np

    by_key = {(x["problem_id"], x["variant"]): x for x in generation_records}
    prediction_by_id = {x["problem_id"]: x for x in predictions}
    labels = []
    for row in rows:
        gold = int(row["gold_answer"])
        accuracies = {}
        extracted = {}
        for variant in RRB_VARIANTS:
            record = by_key[(row["problem_id"], variant)]
            answers = record["extracted_answers"]
            extracted[variant] = answers
            accuracies[variant] = sum(answer == gold for answer in answers) / GENERATION_SAMPLES
        baseline_accuracy = accuracies["baseline"]
        eligible = baseline_accuracy >= 0.5
        max_decay = max(baseline_accuracy - accuracies[name] for name in RRB_VARIANTS[1:])
        robust = bool(max_decay <= 0.0) if eligible else None
        labels.append(
            {
                "problem_id": row["problem_id"],
                "gold_answer": row["gold_answer"],
                "baseline_accuracy": baseline_accuracy,
                "sentence_reversal_accuracy": accuracies["sentence_reversal"],
                "word_reversal_accuracy": accuracies["word_reversal"],
                "split_reversal_accuracy": accuracies["split_reversal"],
                "max_accuracy_decay": max_decay,
                "eligible": eligible,
                "robust_label": robust,
                "extracted_answers": extracted,
            }
        )
    eligible = [x for x in labels if x["eligible"]]
    y = np.asarray([bool(x["robust_label"]) for x in eligible])
    candidate = np.asarray(
        [prediction_by_id[x["problem_id"]]["candidate_prediction"] for x in eligible], dtype=bool
    )
    v6 = np.asarray(
        [prediction_by_id[x["problem_id"]]["v6_prediction"] for x in eligible], dtype=bool
    )
    label_counts = Counter(str(bool(value)) for value in y)
    decision_grade = len(eligible) >= 15 and len(label_counts) == 2 and min(label_counts.values()) >= 4
    failures = []
    if not decision_grade:
        failures.append(
            f"not decision-grade: eligible={len(eligible)}, class_counts={dict(label_counts)}"
        )
    metrics = {
        "eligible_count": len(eligible),
        "label_counts": dict(sorted(label_counts.items())),
        "decision_grade": decision_grade,
    }
    if len(eligible) and len(label_counts) == 2:
        candidate_accuracy = float(np.mean(candidate == y))
        v6_accuracy = float(np.mean(v6 == y))
        candidate_ba = balanced_accuracy(y, candidate)
        v6_ba = balanced_accuracy(y, v6)
        best_constant = max(float(np.mean(y)), float(np.mean(~y)))
        correct_margin = int(np.sum(candidate == y) - np.sum(v6 == y))
        metrics.update(
            {
                "candidate_accuracy": candidate_accuracy,
                "candidate_balanced_accuracy": candidate_ba,
                "v6_accuracy": v6_accuracy,
                "v6_balanced_accuracy": v6_ba,
                "best_constant_accuracy": best_constant,
                "candidate_advantage_over_best_constant": candidate_accuracy - best_constant,
                "candidate_correct_count_margin_vs_v6": correct_margin,
            }
        )
        if candidate_accuracy < best_constant + 0.10:
            failures.append(
                f"candidate accuracy {candidate_accuracy:.6f} < best constant + 0.10 ({best_constant + 0.10:.6f})"
            )
        if candidate_ba < 0.60:
            failures.append(f"candidate balanced accuracy {candidate_ba:.6f} < 0.60")
        if correct_margin < 2:
            failures.append(f"candidate correct-count margin vs V6 {correct_margin} < 2")
    else:
        failures.append("external balanced metrics unavailable because fewer than two classes are present")
    result = {
        "stage": "B",
        "passed": not failures,
        "failures": failures,
        "metrics": metrics,
        "thresholds": {
            "eligible_count_min": 15,
            "each_class_count_min": 4,
            "candidate_accuracy_advantage_over_best_constant_min": 0.10,
            "candidate_balanced_accuracy_min": 0.60,
            "candidate_correct_count_margin_vs_v6_min": 2,
        },
    }
    return result, labels


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_stage_a_report(result: dict) -> None:
    status = "PASS" if result["passed"] else "FAIL"
    failures = "\n".join(f"- {x}" for x in result["failures"]) or "- None"
    metrics = json.dumps(result["metrics"], ensure_ascii=False, indent=2)
    report = f"""# E12 Stage A result

Status: **{status}**

## Metrics

```json
{metrics}
```

## Gate failures

{failures}
"""
    (RESULT_DIR / "STAGE_A_REPORT.md").write_text(report, encoding="utf-8")


def finalize(payload: dict) -> dict:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(RESULT_DIR / "experiment_result.json", payload)
    if FINAL_ZIP.exists():
        FINAL_ZIP.unlink()
    shutil.make_archive(str(FINAL_ZIP.with_suffix("")), "zip", root_dir=WORKING_ROOT, base_dir="result")
    summary = {
        "status": payload["status"],
        "result_zip": str(FINAL_ZIP),
        "result_zip_sha256": sha256_file(FINAL_ZIP),
    }
    write_json(WORKING_ROOT / "FINAL_RESULT.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Download from Kaggle Output: {FINAL_ZIP}", flush=True)
    return summary


def main() -> None:
    import numpy as np

    os.environ["HF_HOME"] = str(MODEL_CACHE)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    WORKING_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_runtime_packages()
    hardware = gpu_preflight()
    checkout_official()
    official_rows, official_audit = prepare_official_rows()
    token = optional_hf_token()
    tokenizer, model, placements = load_exact_model(token)
    official_extraction_audit = extract_diagnostic_views(
        tokenizer, model, "official", official_rows
    )
    official_cube, official_token_counts = load_diagnostic_cube("official", official_rows)
    _, feature_names = make_metacognitive_features(official_cube[:1])
    stage_a = stage_a_validation(official_rows, official_cube, feature_names)
    write_json(RESULT_DIR / "stage_a_result.json", stage_a)
    write_stage_a_report(stage_a)
    manifest = {
        "schema_version": 1,
        "notebook_version": NOTEBOOK_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "weight_dtype": "float16",
        "official_data": official_audit,
        "external_data": None,
        "rrb": None,
        "deployed_v6_control": None,
        "diagnostic_views": DIAGNOSTIC_VIEWS,
        "diagnostic_system_prompt": DIAGNOSTIC_SYSTEM_PROMPT,
        "diagnostic_assistant_prefill": DIAGNOSTIC_PREFILL,
        "layers": LAYERS,
        "feature_names": feature_names,
        "hardware": hardware,
        "model_device_map": placements,
        "official_extraction": official_extraction_audit,
        "external_extraction": None,
        "external_v6_original_extraction": None,
        "max_official_token_count": int(np.max(official_token_counts)),
        "max_external_token_count": None,
        "external_source_loaded_only_after_stage_a_pass": True,
        "external_labels_used_in_stage_a": False,
        "e10_labels_used": False,
    }
    write_json(RESULT_DIR / "run_manifest.json", manifest)

    if not stage_a["passed"]:
        payload = {
            "experiment": "E12_counterbalanced_metacognitive_readout",
            "status": "FAIL_STAGE_A",
            "stage_a": stage_a,
            "stage_b": None,
            "submission_artifact_created": False,
        }
        finalize(payload)
        print("Stage A failed as a scientific negative result; no external labels were generated.")
        return

    external_rows, external_audit = prepare_external_rows(official_rows)
    rrb_audit = validate_transformations(external_rows)
    deployed_v6_artifact, deployed_v6_audit = load_deployed_v6_artifact()
    external_extraction_audit = extract_diagnostic_views(
        tokenizer, model, "external", external_rows
    )
    external_original_audit = extract_external_original_states(tokenizer, model, external_rows)
    external_cube, external_token_counts = load_diagnostic_cube("external", external_rows)
    external_original_cube, external_original_token_counts = load_external_original_cube(
        external_rows
    )
    frozen_predictions = build_external_predictions(
        official_rows,
        official_cube,
        external_rows,
        external_cube,
        external_original_cube,
        deployed_v6_artifact,
    )
    manifest.update(
        {
            "external_data": external_audit,
            "rrb": rrb_audit,
            "deployed_v6_control": deployed_v6_audit,
            "external_extraction": external_extraction_audit,
            "external_v6_original_extraction": external_original_audit,
            "max_external_token_count": int(
                max(np.max(external_token_counts), np.max(external_original_token_counts))
            ),
        }
    )
    write_json(RESULT_DIR / "run_manifest.json", manifest)
    freeze_audit = write_prediction_freeze(frozen_predictions)
    generation_records = generate_external_labels(tokenizer, model, external_rows)
    stage_b, labels = stage_b_validation(external_rows, frozen_predictions, generation_records)
    write_json(RESULT_DIR / "stage_b_result.json", stage_b)
    write_json(RESULT_DIR / "external_labels.json", labels)
    with (RESULT_DIR / "generation_records.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in generation_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    payload = {
        "experiment": "E12_counterbalanced_metacognitive_readout",
        "status": "PASS" if stage_b["passed"] else "FAIL_STAGE_B",
        "stage_a": stage_a,
        "stage_b": stage_b,
        "prediction_freeze": freeze_audit,
        "submission_artifact_created": False,
    }
    finalize(payload)
    print(
        "E12 passed both gates; this permits a separate submission-engineering step."
        if stage_b["passed"]
        else "Stage B failed; V6 remains champion and no submission artifact was created.",
        flush=True,
    )


if __name__ == "__main__":
    main()
