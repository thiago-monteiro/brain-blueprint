from __future__ import annotations

import argparse
import copy
import csv
import glob
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset, DataLoader, Subset

from egomuscle.data.dataset import EgoMuscleDataset, collate_egomuscle, metadata_has_threat
from egomuscle.eval.rdm import compute_rdm
from egomuscle.training.train import EgoMuscleLightningModule, apply_override, load_config


@dataclass
class FractionResult:
    fraction: float
    seed: int
    checkpoint: str
    rsa: float
    gap: float
    n_clips: int
    structure_metrics: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate the threat-correlated sensorimotor threshold experiment."
    )
    parser.add_argument("--mode", choices=("print-train", "train", "prepare-rdm", "analyze", "smoke"), default="analyze")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--config", type=Path, default=Path("egomuscle/training/config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/threat_threshold"))
    parser.add_argument("--fractions", default="0,0.05,0.1,0.2,0.4,0.7,1.0")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--random-seeds", action="store_true", help="Replace --seeds with 30 random seeds")
    parser.add_argument("--checkpoint-pattern", default="checkpoints/threat_f{fraction}_s{seed}/*.ckpt")
    parser.add_argument("--human-threat-rdm", type=Path, default=None)
    parser.add_argument("--eval-manifest", type=Path, default=None, help="CSV with image_path,class rows from build_bold5000_threat_rdm.py.")
    parser.add_argument("--synthetic-human-rdm", action="store_true")
    parser.add_argument("--max-clips", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-max-steps", type=int, default=1200)
    parser.add_argument("--threat-strength", type=float, default=0.75)
    parser.add_argument("--base-video-model", default="OpenGVLab/VideoMAEv2-Base")
    parser.add_argument("--muscle-hidden-dim", type=int, default=256)
    parser.add_argument("--video-trainable-layers", type=int, default=2)
    parser.add_argument(
        "--train-full-cache",
        type=Path,
        default=Path("data/processed/full_cache/train"),
        help="Pre-baked full-frame cache (experiments/bake_full_cache.py).",
    )
    parser.add_argument(
        "--val-full-cache",
        type=Path,
        default=Path("data/processed/full_cache/val"),
        help="Pre-baked full-frame cache for validation.",
    )
    parser.add_argument(
        "--train-num-workers",
        type=int,
        default=12,
        help="DataLoader workers during training (use 0 if cache is missing).",
    )
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


class StaticImageThreatDataset(Dataset):
    def __init__(self, manifest_path: Path, *, n_frames: int, image_size: int) -> None:
        self.rows = []
        with manifest_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                self.rows.append(row)
        if not self.rows:
            raise ValueError(f"No rows in {manifest_path}")
        self.n_frames = int(n_frames)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB").resize((self.image_size, self.image_size))
        arr = np.asarray(image, dtype=np.uint8)
        frame = torch.from_numpy(arr).permute(2, 0, 1)
        frames = frame.unsqueeze(0).repeat(self.n_frames, 1, 1, 1)
        return {
            "frames": frames,
            "muscle": None,
            "activity": row["class"],
            "activity_id": -1,
            "clip_id": Path(row["image_path"]).stem,
            "metadata": dict(row),
        }


def parse_csv_numbers(raw: str, cast: Any) -> list[Any]:
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def build_eval_dataset(args: argparse.Namespace) -> tuple[Subset, list[str], list[str]]:
    if args.eval_manifest is not None:
        dataset = StaticImageThreatDataset(args.eval_manifest, n_frames=args.n_frames, image_size=args.image_size)
        classes = [str(row["class"]) for row in dataset.rows]
        clip_ids = [Path(str(row["image_path"])).stem for row in dataset.rows]
        return Subset(dataset, list(range(len(dataset)))), classes, clip_ids

    datasets: list[EgoMuscleDataset] = []
    classes: list[str] = []
    clip_ids: list[str] = []
    selected: list[int] = []
    offset = 0
    for split in ("val", "test", "train"):
        root = args.dataset_root / split
        if not (root / "clips").exists():
            continue
        dataset = EgoMuscleDataset(
            clip_dir=root / "clips",
            muscle_dir=root / "muscles",
            metadata_path=root / "metadata.json",
            n_frames=args.n_frames,
            image_size=args.image_size,
            muscle_dim=None,
            require_muscle=False,
            frame_cache_dir=root / "frames",
            write_frame_cache=False,
        )
        datasets.append(dataset)
        for local_idx, record in enumerate(dataset.records):
            cls = "threat" if metadata_has_threat(record.metadata, record.activity) else "neutral"
            selected.append(offset + local_idx)
            classes.append(cls)
            clip_ids.append(record.video_path.stem)
        offset += len(dataset)
    if not datasets:
        raise FileNotFoundError(f"No processed split clips found under {args.dataset_root}")
    if args.max_clips and len(selected) > args.max_clips:
        rng = np.random.default_rng(0)
        threat_idx = [idx for idx, cls in enumerate(classes) if cls == "threat"]
        neutral_idx = [idx for idx, cls in enumerate(classes) if cls == "neutral"]
        per_class = max(1, args.max_clips // 2)
        chosen_local = []
        for pool in (threat_idx, neutral_idx):
            if pool:
                take = min(per_class, len(pool))
                chosen_local.extend(rng.choice(pool, size=take, replace=False).tolist())
        chosen_local = sorted(chosen_local)[: args.max_clips]
        selected = [selected[i] for i in chosen_local]
        classes = [classes[i] for i in chosen_local]
        clip_ids = [clip_ids[i] for i in chosen_local]
    if len(selected) < 4:
        raise ValueError("Threat RSA requires at least four clips.")
    if len(set(classes)) < 2:
        raise ValueError("Threat RSA requires both threat and neutral clips.")
    return Subset(ConcatDataset(datasets), selected), classes, clip_ids


def synthetic_human_threat_rdm(classes: list[str]) -> np.ndarray:
    labels = np.asarray(classes)
    rdm = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for i in range(len(labels)):
        for j in range(len(labels)):
            rdm[i, j] = 0.25 if labels[i] == labels[j] else 1.0
    np.fill_diagonal(rdm, 0.0)
    return rdm


def load_human_rdm(args: argparse.Namespace, classes: list[str]) -> np.ndarray:
    if args.synthetic_human_rdm or args.mode == "smoke":
        return synthetic_human_threat_rdm(classes)
    if args.human_threat_rdm is None:
        raise ValueError(
            "Primary analysis requires --human-threat-rdm. Use --synthetic-human-rdm only for smoke tests."
        )
    rdm = np.load(args.human_threat_rdm)
    if rdm.shape != (len(classes), len(classes)):
        raise ValueError(f"Human RDM shape {rdm.shape} does not match {len(classes)} selected clips.")
    return rdm


def load_module(checkpoint: Path, config_path: Path, overrides: list[str], device: torch.device) -> EgoMuscleLightningModule:
    payload = torch.load(checkpoint, map_location=device)
    config = payload.get("hyper_parameters")
    if not isinstance(config, dict):
        config = load_config(config_path)
    else:
        config = copy.deepcopy(config)
    for override in overrides:
        apply_override(config, override)
    module = EgoMuscleLightningModule(config)
    state_dict = payload.get("state_dict", payload)
    candidates = [
        state_dict,
        {key.replace("model._orig_mod.", "model."): value for key, value in state_dict.items()},
    ]
    last_error: RuntimeError | None = None
    for candidate in candidates:
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
    from egomuscle.data.dataset import IMAGE_MEAN, IMAGE_STD

    moved = dict(batch)
    frames = moved["frames"].to(device)
    if frames.dtype == torch.uint8:
        frames = frames.float().div_(255.0)
        frames = (frames - IMAGE_MEAN.to(device)) / IMAGE_STD.to(device)
    moved["frames"] = frames
    moved["activity_id"] = moved["activity_id"].to(device)
    if torch.is_tensor(moved.get("muscle")):
        moved["muscle"] = moved["muscle"].to(device)
    return moved


def attention_entropy(attention: torch.Tensor | None) -> float:
    if attention is None:
        return float("nan")
    probs = attention.detach().float().clamp_min(1.0e-8)
    entropy = -(probs * probs.log()).sum(dim=-1).mean()
    return float(entropy.cpu())


def collect_model_rdm(
    module: EgoMuscleLightningModule,
    dataset: Subset,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_egomuscle,
        pin_memory=(device.type == "cuda"),
    )
    reps: list[torch.Tensor] = []
    metrics: dict[str, list[float]] = {
        "pooled_norm": [],
        "video_repr_norm": [],
        "conditioning_repr_norm": [],
        "attention_entropy": [],
        "fast_state_norm": [],
        "pbit_entropy": [],
    }
    with torch.no_grad():
        for batch in loader:
            batch = transfer_batch(batch, device)
            outputs = module.model(
                frames=batch["frames"],
                muscle=batch.get("muscle"),
                activity_ids=batch["activity_id"],
                mask_ratio=0.0,
            )
            reps.append(outputs.pooled.detach().cpu())
            metrics["pooled_norm"].append(float(outputs.pooled.detach().norm(dim=-1).mean().cpu()))
            if outputs.video_repr is not None:
                metrics["video_repr_norm"].append(float(outputs.video_repr.detach().norm(dim=-1).mean().cpu()))
            if outputs.conditioning_repr is not None:
                metrics["conditioning_repr_norm"].append(float(outputs.conditioning_repr.detach().norm(dim=-1).mean().cpu()))
            metrics["attention_entropy"].append(attention_entropy(outputs.attention))
            if outputs.fast_state is not None:
                metrics["fast_state_norm"].append(float(outputs.fast_state.detach().norm(dim=-1).mean().cpu()))
            if outputs.pbit_entropy is not None:
                metrics["pbit_entropy"].append(float(outputs.pbit_entropy.detach().mean().cpu()))
    summary = {}
    for key, values in metrics.items():
        clean = [v for v in values if math.isfinite(v)]
        summary[key] = float(np.mean(clean)) if clean else float("nan")
    return compute_rdm(torch.cat(reps, dim=0).numpy()), summary


def rdm_rsa(model_rdm: np.ndarray, human_rdm: np.ndarray) -> float:
    tri = np.triu_indices(model_rdm.shape[0], k=1)
    rho = spearmanr(model_rdm[tri], human_rdm[tri]).statistic
    return float(0.0 if np.isnan(rho) else rho)


def resolve_checkpoint(pattern: str, fraction: float, seed: int) -> Path:
    key = f"{fraction:g}".replace(".", "p")
    candidates = sorted(glob.glob(pattern.format(fraction=key, seed=seed)))
    if not candidates:
        candidates = sorted(glob.glob(pattern.format(fraction=f"{fraction:g}", seed=seed)))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint for fraction={fraction:g} seed={seed}: {pattern}")
    return Path(candidates[-1])


def build_train_command(args: argparse.Namespace, fraction: float, seed: int) -> list[str]:
    key = f"{fraction:g}".replace(".", "p")
    run_name = f"threat_f{key}_s{seed}"
    cmd = [
        sys.executable,
        "-m",
        "egomuscle.training.train",
        "--config",
        str(args.config),
        "--override",
        f"seed={seed}",
        "--override",
        f"logging.run_name={run_name}",
        "--override",
        "logging.tags=[threat_threshold]",
        "--override",
        f"model.video_model_name={args.base_video_model}",
        "--override",
        "model.video_trainable_strategy=last_n",
        "--override",
        f"model.video_trainable_layers={args.video_trainable_layers}",
        "--override",
        "model.video_unfreeze_embeddings=false",
        "--override",
        f"model.muscle_hidden_dim={args.muscle_hidden_dim}",
        "--override",
        "model.fusion_mode=cross_attn",
        "--override",
        f"training.max_steps={args.train_max_steps}",
        "--override",
        f"training.max_epochs={max(20, (int(args.train_max_steps) + 16) // 17 + 2)}",
        "--override",
        "training.early_stopping_patience=null",
        "--override",
        "data.temporal_sample_mode=random_stride",
        "--override",
        "data.train.replacement_sampling=true",
        "--override",
        f"data.train.threat_correlation_fraction={fraction}",
        "--override",
        f"data.train.threat_signature_strength={args.threat_strength}",
        "--override",
        f"data.train.threat_seed={seed}",
        "--override",
        "data.train.write_frame_cache=false",
        "--override",
        f"data.train.full_cache_dir={args.train_full_cache}",
        "--override",
        f"data.val.full_cache_dir={args.val_full_cache}",
        "--override",
        f"data.num_workers={args.train_num_workers}",
        "--override",
        f"training.compile={'true' if os.environ.get('THREAT_COMPILE', '1') == '1' else 'false'}",
        "--override",
        f"training.compile_mode={os.environ.get('THREAT_COMPILE_MODE', 'default')}",
    ]
    for override in args.override:
        cmd.extend(["--override", override])
    return cmd


def detect_threshold(results: list[FractionResult]) -> dict[str, Any]:
    by_fraction: dict[float, list[FractionResult]] = {}
    for result in results:
        by_fraction.setdefault(result.fraction, []).append(result)
    points = []
    for fraction in sorted(by_fraction):
        gaps = [row.gap for row in by_fraction[fraction]]
        rsas = [row.rsa for row in by_fraction[fraction]]
        points.append(
            {
                "fraction": fraction,
                "gap_mean": float(np.mean(gaps)),
                "gap_std": float(np.std(gaps)),
                "rsa_mean": float(np.mean(rsas)),
                "num_seeds": len(gaps),
            }
        )
    if len(points) < 2:
        return {"points": points, "threshold_fraction": None, "phase_transition_candidate": False}
    gaps = np.asarray([p["gap_mean"] for p in points], dtype=np.float64)
    drops = gaps[:-1] - gaps[1:]
    max_drop_idx = int(np.argmax(drops))
    total_closure = float(gaps[0] - gaps[-1])
    snap_ratio = float(drops[max_drop_idx] / total_closure) if total_closure > 0 else 0.0
    threshold_fraction = float(points[max_drop_idx + 1]["fraction"])
    phase = bool(total_closure > 0 and snap_ratio >= 0.5)
    structures = {}
    threshold_rows = [row for row in results if row.fraction == threshold_fraction]
    if threshold_rows:
        keys = sorted({key for row in threshold_rows for key in row.structure_metrics})
        for key in keys:
            values = [row.structure_metrics[key] for row in threshold_rows if math.isfinite(row.structure_metrics.get(key, float("nan")))]
            structures[key] = float(np.mean(values)) if values else float("nan")
    return {
        "points": points,
        "threshold_fraction": threshold_fraction,
        "largest_gap_drop": float(drops[max_drop_idx]),
        "total_gap_closure": total_closure,
        "snap_ratio": snap_ratio,
        "phase_transition_candidate": phase,
        "structures_active_at_threshold": structures,
    }


def run_analyze(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fractions = parse_csv_numbers(args.fractions, float)
    seeds = parse_csv_numbers(args.seeds, int)
    dataset, classes, clip_ids = build_eval_dataset(args)
    eval_payload = {
        "clip_ids": clip_ids,
        "classes": classes,
        "rdm_shape": [len(classes), len(classes)],
        "human_rdm_required_for_primary_analysis": not args.synthetic_human_rdm,
    }
    (args.output_dir / "eval_clips.json").write_text(json.dumps(eval_payload, indent=2))
    if args.mode == "prepare-rdm":
        template = np.zeros((len(classes), len(classes)), dtype=np.float32)
        np.save(args.output_dir / "human_threat_rdm_template.npy", template)
        print(json.dumps(eval_payload, indent=2))
        return
    human_rdm = load_human_rdm(args, classes)
    np.save(args.output_dir / "human_threat_rdm_used.npy", human_rdm)
    device = torch.device(args.device)

    results: list[FractionResult] = []
    for fraction in fractions:
        for seed in seeds:
            checkpoint = resolve_checkpoint(args.checkpoint_pattern, fraction, seed)
            module = load_module(checkpoint, args.config, args.override, device)
            model_rdm, structure_metrics = collect_model_rdm(module, dataset, args, device)
            rsa = rdm_rsa(model_rdm, human_rdm)
            gap = 1.0 - rsa
            key = f"f{fraction:g}_s{seed}".replace(".", "p")
            np.save(args.output_dir / f"{key}_model_rdm.npy", model_rdm)
            result = FractionResult(
                fraction=fraction,
                seed=seed,
                checkpoint=str(checkpoint),
                rsa=rsa,
                gap=gap,
                n_clips=len(classes),
                structure_metrics=structure_metrics,
            )
            results.append(result)
            (args.output_dir / f"{key}.json").write_text(json.dumps(result.__dict__, indent=2))
    summary = {
        "question_tracking": {
            "does_gap_close": None,
            "is_closure_non_linear": None,
            "structures_active_at_threshold": None,
        },
        "human_rdm": str(args.human_threat_rdm) if args.human_threat_rdm else "synthetic_smoke_target",
        "results": [row.__dict__ for row in results],
        "threshold": detect_threshold(results),
    }
    points = summary["threshold"]["points"]
    if len(points) >= 2:
        summary["question_tracking"]["does_gap_close"] = bool(points[-1]["gap_mean"] < points[0]["gap_mean"])
        summary["question_tracking"]["is_closure_non_linear"] = bool(summary["threshold"]["phase_transition_candidate"])
        summary["question_tracking"]["structures_active_at_threshold"] = summary["threshold"]["structures_active_at_threshold"]
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def run_smoke(args: argparse.Namespace) -> None:
    args.synthetic_human_rdm = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    classes = ["threat", "threat", "neutral", "neutral", "threat", "neutral"]
    human_rdm = synthetic_human_threat_rdm(classes)
    rng = np.random.default_rng(0)
    results: list[FractionResult] = []
    for fraction in parse_csv_numbers(args.fractions, float)[:3]:
        for seed in parse_csv_numbers(args.seeds, int)[:2]:
            noise = 1.0 - fraction
            model_rdm = human_rdm + rng.normal(0, 0.2 * noise, size=human_rdm.shape)
            model_rdm = (model_rdm + model_rdm.T) / 2.0
            np.fill_diagonal(model_rdm, 0.0)
            rsa = rdm_rsa(model_rdm, human_rdm)
            results.append(
                FractionResult(
                    fraction=fraction,
                    seed=seed,
                    checkpoint="synthetic",
                    rsa=rsa,
                    gap=1.0 - rsa,
                    n_clips=len(classes),
                    structure_metrics={"pooled_norm": float(1.0 + fraction), "attention_entropy": float(0.5 + fraction)},
                )
            )
    summary = {"results": [row.__dict__ for row in results], "threshold": detect_threshold(results)}
    (args.output_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    if getattr(args, "random_seeds", False):
        import random

        args.seeds = ",".join(str(random.randint(0, 2**31 - 1)) for _ in range(30))
    if args.mode in {"print-train", "train"}:
        for fraction in parse_csv_numbers(args.fractions, float):
            for seed in parse_csv_numbers(args.seeds, int):
                cmd = build_train_command(args, fraction, seed)
                if args.mode == "print-train":
                    print(" ".join(cmd))
                else:
                    subprocess.run(cmd, cwd=ROOT, check=True)
        return
    if args.mode == "smoke":
        run_smoke(args)
        return
    run_analyze(args)


if __name__ == "__main__":
    main()
