from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from egomuscle.data.dataset import EgoMuscleDataset, collate_egomuscle
from egomuscle.eval.rdm import compute_rdm
from egomuscle.training.train import EgoMuscleLightningModule, apply_override, load_config


RELEVANT_KEYWORDS = {
    "fall",
    "slip",
    "trip",
    "stumble",
    "collapse",
    "land",
    "landing",
    "jump",
    "leap",
    "squat",
    "lunge",
    "lift",
    "carry",
    "push",
    "pull",
    "climb",
    "crawl",
    "kick",
    "punch",
    "hit",
    "fight",
    "dodge",
    "duck",
    "avoid",
    "protect",
}

NEUTRAL_KEYWORDS = {
    "walk",
    "walking",
    "run",
    "running",
    "jog",
    "jogging",
    "step",
    "stepping",
    "turn",
    "turning",
    "stand",
    "standing",
    "locomotion",
    "pace",
    "pacing",
}


E2_OVERRIDES = [
    "model.video_model_name=MCG-NJU/videomae-base",
    "model.use_video=true",
    "model.use_muscle=true",
    "model.fusion_mode=late",
    "training.compile=true",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the E2 viability-boundary RSA experiment without retraining.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=[],
        help="Checkpoint file or directory. May be repeated. Directories are searched for *.ckpt.",
    )
    parser.add_argument(
        "--checkpoint-glob",
        default="experiments/results/clean_runs/paper_clean_v1/ablations_epochs100/seed_*/E2/checkpoints/E2_late_fusion/*.ckpt",
    )
    parser.add_argument("--config", type=Path, default=Path("egomuscle/training/config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/viability_boundary_e2"))
    parser.add_argument("--mint-manifest", type=Path, default=Path("data/processed/manifests/mint_sequences.jsonl"))
    parser.add_argument("--max-clips-per-partition", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def flatten_strings(value: Any) -> list[str]:
    strings: list[str] = []
    if value is None:
        return strings
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return []
    if isinstance(value, dict):
        for key, child in value.items():
            key_lower = str(key).lower()
            if any(token in key_lower for token in ("activity", "label", "action", "act_cat", "proc_label", "babel", "ann")):
                strings.extend(flatten_strings(child))
            elif isinstance(child, (dict, list, tuple)):
                strings.extend(flatten_strings(child))
        return strings
    if isinstance(value, (list, tuple)):
        for child in value:
            strings.extend(flatten_strings(child))
    return strings


def keyword_hits(labels: list[str], keywords: set[str]) -> set[str]:
    text = " ".join(labels).lower().replace("_", " ")
    tokens = set(re.findall(r"[a-z]+", text))
    hits = set()
    for keyword in keywords:
        normalized = keyword.lower()
        if " " in normalized:
            if normalized in text:
                hits.add(keyword)
        elif normalized in tokens:
            hits.add(keyword)
    return hits


def classify_metadata(
    metadata: dict[str, Any],
    babel_by_mint_key: dict[str, dict[str, Any]] | None = None,
) -> tuple[str | None, list[str], set[str], set[str], str]:
    if babel_by_mint_key and metadata.get("mint_key") in babel_by_mint_key:
        metadata = {**metadata, "babel": babel_by_mint_key[str(metadata["mint_key"])]}
    labels = sorted(set(label.strip() for label in flatten_strings(metadata) if label and label.strip()))
    if not labels:
        return None, labels, set(), set(), "missing_babel_labels"
    relevant = keyword_hits(labels, RELEVANT_KEYWORDS)
    neutral = keyword_hits(labels, NEUTRAL_KEYWORDS)
    if relevant and neutral:
        return None, labels, relevant, neutral, "ambiguous_both_classes"
    if relevant:
        return "viability_relevant", labels, relevant, neutral, "included"
    if neutral:
        return "viability_neutral", labels, relevant, neutral, "included"
    return None, labels, relevant, neutral, "no_keyword_match"


def load_babel_manifest_index(manifest_path: Path) -> dict[str, dict[str, Any]]:
    if not manifest_path.exists():
        return {}
    index: dict[str, dict[str, Any]] = {}
    with manifest_path.open("r") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            mint_key = record.get("mint_key")
            babel = record.get("babel")
            if mint_key and babel:
                index[str(mint_key)] = babel
    return index


def build_split_dataset(dataset_root: Path, split: str, args: argparse.Namespace) -> EgoMuscleDataset | None:
    split_root = dataset_root / split
    clip_dir = split_root / "clips"
    if not clip_dir.exists():
        return None
    try:
        return EgoMuscleDataset(
            clip_dir=clip_dir,
            muscle_dir=split_root / "muscles",
            metadata_path=split_root / "metadata.json",
            n_frames=args.n_frames,
            image_size=args.image_size,
            muscle_dim=None,
            require_muscle=False,
            frame_cache_dir=split_root / "frames",
            write_frame_cache=False,
        )
    except ValueError as exc:
        if "No video records found" in str(exc):
            return None
        raise


def build_partition(args: argparse.Namespace) -> tuple[ConcatDataset, list[dict[str, Any]], list[int]]:
    datasets = []
    candidates: list[dict[str, Any]] = []
    babel_by_mint_key = load_babel_manifest_index(args.mint_manifest)
    offset = 0
    for split in ("train", "val", "test"):
        dataset = build_split_dataset(args.dataset_root, split, args)
        if dataset is None:
            continue
        datasets.append(dataset)
        for idx, record in enumerate(dataset.records):
            cls, labels, relevant_hits, neutral_hits, reason = classify_metadata(record.metadata or {}, babel_by_mint_key)
            candidates.append(
                {
                    "global_index": offset + idx,
                    "split": split,
                    "clip_id": record.video_path.stem,
                    "class": cls or "",
                    "included": cls is not None,
                    "reason": reason,
                    "labels": "|".join(labels),
                    "relevant_hits": "|".join(sorted(relevant_hits)),
                    "neutral_hits": "|".join(sorted(neutral_hits)),
                    "video_path": str(record.video_path),
                }
            )
        offset += len(dataset)

    if not datasets:
        raise FileNotFoundError(f"No non-empty split clip directories found under {args.dataset_root}")

    selected_indices: list[int] = []
    rng = np.random.default_rng(args.seed)
    remaining_per_class = args.max_clips_per_partition
    for split in ("train", "val", "test"):
        relevant = [row for row in candidates if row["split"] == split and row["class"] == "viability_relevant"]
        neutral = [row for row in candidates if row["split"] == split and row["class"] == "viability_neutral"]
        n = min(len(relevant), len(neutral))
        if remaining_per_class is not None:
            n = min(n, remaining_per_class)
        if n == 0:
            continue
        rel_pick = rng.choice(len(relevant), size=n, replace=False)
        neu_pick = rng.choice(len(neutral), size=n, replace=False)
        for idx in rel_pick:
            selected_indices.append(int(relevant[int(idx)]["global_index"]))
        for idx in neu_pick:
            selected_indices.append(int(neutral[int(idx)]["global_index"]))
        if remaining_per_class is not None:
            remaining_per_class -= n
            if remaining_per_class <= 0:
                break

    selected_set = set(selected_indices)
    for row in candidates:
        row["selected"] = int(row["global_index"] in selected_set)

    if not selected_indices:
        raise ValueError("No balanced viability partitions could be selected from metadata labels.")
    return ConcatDataset(datasets), candidates, selected_indices


def write_partition_manifest(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "global_index",
        "split",
        "clip_id",
        "class",
        "included",
        "selected",
        "reason",
        "labels",
        "relevant_hits",
        "neutral_hits",
        "video_path",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def resolve_checkpoints(args: argparse.Namespace) -> list[Path]:
    checkpoints: list[Path] = []
    for path in args.checkpoint:
        if path.is_dir():
            checkpoints.extend(sorted(path.glob("*.ckpt")))
        else:
            checkpoints.append(path)
    if not checkpoints:
        checkpoints = sorted(Path(".").glob(args.checkpoint_glob))
    checkpoints = [path for path in checkpoints if path.exists()]
    if not checkpoints:
        raise FileNotFoundError("No E2 checkpoints found. Pass --checkpoint or adjust --checkpoint-glob.")
    return checkpoints


def infer_seed(path: Path, default_seed: int) -> int:
    for part in path.parts:
        match = re.fullmatch(r"seed_(\d+)", part)
        if match:
            return int(match.group(1))
    return default_seed


def load_e2_module(checkpoint_path: Path, config_path: Path, overrides: list[str], device: torch.device) -> EgoMuscleLightningModule:
    payload = torch.load(checkpoint_path, map_location=device)
    config = payload.get("hyper_parameters")
    if not isinstance(config, dict):
        config = load_config(config_path)
    else:
        config = copy.deepcopy(config)
    for override in overrides:
        apply_override(config, override)
    config.setdefault("training", {})["compile"] = config.get("training", {}).get("compile", True)

    module = EgoMuscleLightningModule(config)
    state_dict = payload.get("state_dict", payload)
    candidate_state_dicts: list[dict[str, Any]] = [state_dict]
    stripped_state_dict = {key.replace("model._orig_mod.", "model."): value for key, value in state_dict.items()}
    if stripped_state_dict != state_dict:
        candidate_state_dicts.append(stripped_state_dict)
    prefixed_state_dict = {
        (key.replace("model.", "model._orig_mod.", 1) if key.startswith("model.") and not key.startswith("model._orig_mod.") else key): value
        for key, value in state_dict.items()
    }
    if prefixed_state_dict != state_dict:
        candidate_state_dicts.append(prefixed_state_dict)

    last_error: RuntimeError | None = None
    for candidate in candidate_state_dicts:
        try:
            module.load_state_dict(candidate, strict=False)
            last_error = None
            break
        except RuntimeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error

    module.to(device)
    module.eval()
    return module


def transfer_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    frames = moved["frames"].to(device, non_blocking=True)
    if frames.dtype == torch.uint8:
        from egomuscle.data.dataset import IMAGE_MEAN, IMAGE_STD

        frames = frames.float().div_(255.0)
        frames = (frames - IMAGE_MEAN.to(device)) / IMAGE_STD.to(device)
    moved["frames"] = frames
    moved["activity_id"] = moved["activity_id"].to(device, non_blocking=True)
    moved["muscle"] = None
    return moved


def collect_representations(
    module: Any,
    dataloader: DataLoader,
    device: torch.device,
    class_by_clip_id: dict[str, str],
) -> tuple[np.ndarray, list[str], list[str]]:
    features: list[torch.Tensor] = []
    classes: list[str] = []
    clip_ids: list[str] = []
    module.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = transfer_batch(batch, device)
            outputs = module.model(
                frames=batch["frames"],
                muscle=None,
                activity_ids=batch["activity_id"],
                mask_ratio=0.0,
            )
            features.append(outputs.pooled.detach().cpu())
            clip_ids.extend(batch["clip_id"])
            classes.extend(class_by_clip_id.get(clip_id, "") for clip_id in batch["clip_id"])
    return torch.cat(features, dim=0).numpy(), classes, clip_ids


def mean_distance(rdm: np.ndarray, rows: np.ndarray, cols: np.ndarray, *, within: bool) -> float:
    if within:
        if len(rows) < 2:
            return float("nan")
        sub = rdm[np.ix_(rows, rows)]
        tri = np.triu_indices(len(rows), k=1)
        return float(sub[tri].mean())
    return float(rdm[np.ix_(rows, cols)].mean())


def boundary_metrics(rdm: np.ndarray, classes: list[str]) -> dict[str, float]:
    relevant = np.array([idx for idx, cls in enumerate(classes) if cls == "viability_relevant"], dtype=np.int64)
    neutral = np.array([idx for idx, cls in enumerate(classes) if cls == "viability_neutral"], dtype=np.int64)
    within_relevant = mean_distance(rdm, relevant, relevant, within=True)
    within_neutral = mean_distance(rdm, neutral, neutral, within=True)
    across = mean_distance(rdm, relevant, neutral, within=False)
    effect = across - float(np.nanmean([within_relevant, within_neutral]))
    return {
        "num_viability_relevant": int(len(relevant)),
        "num_viability_neutral": int(len(neutral)),
        "within_relevant_distance": within_relevant,
        "within_neutral_distance": within_neutral,
        "across_boundary_distance": across,
        "model_boundary_effect": effect,
    }


def permutation_null(
    rdm: np.ndarray,
    classes: list[str],
    *,
    n_permutations: int,
    seed: int,
) -> dict[str, Any]:
    observed = boundary_metrics(rdm, classes)["model_boundary_effect"]
    rng = np.random.default_rng(seed)
    labels = np.array(classes, dtype=object)
    null_effects = np.empty(n_permutations, dtype=np.float64)
    for idx in range(n_permutations):
        permuted = rng.permutation(labels)
        null_effects[idx] = boundary_metrics(rdm, permuted.tolist())["model_boundary_effect"]

    centered = null_effects - float(null_effects.mean())
    observed_centered = abs(float(observed) - float(null_effects.mean()))
    p_two_sided = (float(np.sum(np.abs(centered) >= observed_centered)) + 1.0) / (float(n_permutations) + 1.0)
    p_greater = (float(np.sum(null_effects >= observed)) + 1.0) / (float(n_permutations) + 1.0)
    std = float(null_effects.std(ddof=1)) if n_permutations > 1 else float("nan")
    z = (float(observed) - float(null_effects.mean())) / std if std > 0 else float("nan")
    return {
        "n_permutations": int(n_permutations),
        "permutation_null_mean": float(null_effects.mean()),
        "permutation_null_sd": std,
        "permutation_null_ci": [
            float(np.quantile(null_effects, 0.025)),
            float(np.quantile(null_effects, 0.975)),
        ],
        "permutation_p_greater": p_greater,
        "permutation_p_two_sided": p_two_sided,
        "permutation_z": z,
    }


def bootstrap_ci(values: list[float], *, n_bootstrap: int, seed: int) -> list[float]:
    clean = np.array([value for value in values if math.isfinite(value)], dtype=np.float64)
    if len(clean) == 0:
        return [float("nan"), float("nan")]
    if len(clean) == 1:
        return [float(clean[0]), float(clean[0])]
    rng = np.random.default_rng(seed)
    samples = rng.choice(clean, size=(n_bootstrap, len(clean)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def write_distances_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fields = [
        "seed",
        "checkpoint",
        "num_viability_relevant",
        "num_viability_neutral",
        "within_relevant_distance",
        "within_neutral_distance",
        "across_boundary_distance",
        "model_boundary_effect",
        "permutation_null_mean",
        "permutation_null_sd",
        "permutation_p_greater",
        "permutation_p_two_sided",
        "permutation_z",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_effect_plot(seed_results: list[dict[str, Any]], output_path: Path) -> None:
    effects = [float(row["model_boundary_effect"]) for row in seed_results]
    seeds = [str(row["seed"]) for row in seed_results]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.bar(seeds, effects, color="#4C78A8")
    ax.set_xlabel("E2 seed")
    ax.set_ylabel("Across - mean(within) cosine distance")
    ax.set_title("Viability Boundary Effect")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dataset, partition_rows, selected_indices = build_partition(args)
    write_partition_manifest(partition_rows, args.output_dir / "partition_manifest.csv")

    subset = Subset(dataset, selected_indices)
    class_by_clip_id = {row["clip_id"]: row["class"] for row in partition_rows if row["selected"]}
    dataloader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_egomuscle,
    )

    checkpoints = resolve_checkpoints(args)
    seed_results: list[dict[str, Any]] = []

    for default_seed, checkpoint in enumerate(checkpoints):
        seed = infer_seed(checkpoint, default_seed)
        module = load_e2_module(checkpoint, args.config, [*E2_OVERRIDES, *args.override], device)
        features, classes, clip_ids = collect_representations(module, dataloader, device, class_by_clip_id)
        rdm = compute_rdm(features)
        metrics = boundary_metrics(rdm, classes)
        null_metrics = permutation_null(
            rdm,
            classes,
            n_permutations=args.n_permutations,
            seed=args.seed + seed,
        )
        payload = {
            "seed": seed,
            "checkpoint": str(checkpoint),
            "dataset_root": str(args.dataset_root),
            "clip_ids": clip_ids,
            "classes": classes,
            "rdm_shape": list(rdm.shape),
            **metrics,
            **null_metrics,
        }
        np.save(args.output_dir / f"seed_{seed}_rdm.npy", rdm)
        (args.output_dir / f"seed_{seed}.json").write_text(json.dumps(payload, indent=2))
        seed_results.append(payload)

    seed_results = sorted(seed_results, key=lambda row: int(row["seed"]))
    write_distances_csv(seed_results, args.output_dir / "viability_boundary_distances.csv")
    write_effect_plot(seed_results, args.output_dir / "viability_boundary_effect.pdf")

    effects = [float(row["model_boundary_effect"]) for row in seed_results]
    model_mean = float(np.mean(effects))
    model_ci = bootstrap_ci(effects, n_bootstrap=args.n_bootstrap, seed=args.seed)
    permutation_p_values = [float(row["permutation_p_greater"]) for row in seed_results]
    permutation_z_values = [float(row["permutation_z"]) for row in seed_results if math.isfinite(float(row["permutation_z"]))]
    selected_by_class = defaultdict(int)
    for row in partition_rows:
        if row["selected"]:
            selected_by_class[row["class"]] += 1

    summary = {
        "model": "E2_late_fusion",
        "checkpoints": [str(path) for path in checkpoints],
        "dataset_root": str(args.dataset_root),
        "num_viability_relevant": int(selected_by_class["viability_relevant"]),
        "num_viability_neutral": int(selected_by_class["viability_neutral"]),
        "seed_results": seed_results,
        "model_boundary_effect_mean": model_mean,
        "model_boundary_effect_ci": model_ci,
        "permutation_null": {
            "n_permutations": int(args.n_permutations),
            "mean_p_greater": float(np.mean(permutation_p_values)) if permutation_p_values else None,
            "min_p_greater": float(np.min(permutation_p_values)) if permutation_p_values else None,
            "max_p_greater": float(np.max(permutation_p_values)) if permutation_p_values else None,
            "mean_z": float(np.mean(permutation_z_values)) if permutation_z_values else None,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
