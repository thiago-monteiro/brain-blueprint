from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from egomuscle.data.dataset import EgoMuscleDataset, collate_egomuscle
from egomuscle.eval.twente_eval import load_lightning_module
from egomuscle.training.train import apply_override, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gap-retention and uncertainty probes for checkpoints.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("egomuscle/training/config.yaml"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--gaps", nargs="+", type=int, default=[0, 2, 4, 8, 12])
    parser.add_argument("--nback", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--motion-clusters", type=int, default=16)
    parser.add_argument("--mask-video", action="store_true")
    parser.add_argument("--mask-muscle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/memory_probe_suite.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-probe-batches", type=int)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def build_dataset(config: dict[str, Any], dataset_root: Path, split: str) -> EgoMuscleDataset:
    split_cfg = config["data"][split]
    return EgoMuscleDataset(
        clip_dir=dataset_root / split / "clips",
        muscle_dir=dataset_root / split / "muscles",
        metadata_path=dataset_root / split / "metadata.json",
        n_frames=int(config["data"].get("n_frames", 16)),
        image_size=int(config["data"].get("image_size", 224)),
        muscle_dim=config["data"].get("muscle_dim"),
        require_muscle=True,
        scramble_video=bool(split_cfg.get("scramble_video", False)),
        temporal_sample_mode="sparse_uniform",
        muscle_time_offset=0,
        muscle_noise_std=0.0,
        frame_cache_dir=None,
        full_cache_dir=split_cfg.get("full_cache_dir"),
        write_frame_cache=False,
        is_train=False,
    )


def mask_gap(
    frames: torch.Tensor | None,
    muscle: torch.Tensor | None,
    *,
    gap: int,
    mask_ratio: float,
    mask_video: bool,
    mask_muscle: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if gap <= 0:
        return frames, muscle
    seq_len = frames.shape[1] if frames is not None else muscle.shape[1]
    t_split = min(max(int(seq_len * (1.0 - mask_ratio)), 1), seq_len - 1)
    start = max(0, t_split - gap)
    end = t_split
    if mask_video and frames is not None:
        frames = frames.clone()
        frames[:, start:end] = 0
    if mask_muscle and muscle is not None:
        muscle = muscle.clone()
        muscle[:, start:end] = 0
    return frames, muscle


def fit_retention_tau(rows: list[dict[str, float]]) -> float | None:
    gaps = np.asarray([row["gap"] for row in rows], dtype=np.float64)
    performance = -np.asarray([row["mse"] for row in rows], dtype=np.float64)
    if len(gaps) < 3 or np.allclose(performance, performance[0]):
        return None
    floor = performance.min() - 1e-6
    y = np.log(np.maximum(performance - floor, 1e-8))
    slope, _ = np.polyfit(gaps, y, 1)
    if slope >= 0:
        return None
    return float(-1.0 / slope)


def trapezoid_auc(y: list[float], x: list[float]) -> float:
    fn = getattr(np, "trapezoid", None)
    if fn is None:
        fn = np.trapz
    return float(fn(y, x))


def evaluate_gap(
    module,
    loader: DataLoader,
    device: torch.device,
    *,
    gap: int,
    args: argparse.Namespace,
    mask_ratio: float,
    max_batches: int | None,
) -> dict[str, float]:
    mses: list[float] = []
    module.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            frames = batch["frames"].to(device)
            muscle = batch["muscle"].to(device)
            activity_ids = batch["activity_id"].to(device)
            frames, muscle = mask_gap(
                frames,
                muscle,
                gap=gap,
                mask_ratio=mask_ratio,
                mask_video=args.mask_video,
                mask_muscle=args.mask_muscle,
            )
            outputs = module.model(frames=frames, muscle=muscle, activity_ids=activity_ids, mask_ratio=mask_ratio)
            if outputs.pred is None or outputs.target is None:
                raise ValueError("Memory probe requires muscle predictions.")
            mses.append(float(F.mse_loss(outputs.pred, outputs.target).detach().cpu().item()))
    row = {"gap": float(gap), "mse": float(np.mean(mses))}
    return row


def linear_accuracy(features: np.ndarray, labels: np.ndarray) -> float | None:
    valid = labels >= 0
    features = features[valid]
    labels = labels[valid]
    if len(labels) < 8 or len(np.unique(labels)) < 2:
        return None
    try:
        train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.35, random_state=0, stratify=labels)
    except ValueError:
        train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.35, random_state=0)
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_x, train_y)
    return float(accuracy_score(test_y, clf.predict(test_x)))


def collect_memory_features(module, loader: DataLoader, device: torch.device, *, max_batches: int | None) -> dict[str, np.ndarray]:
    features: list[np.ndarray] = []
    muscles: list[np.ndarray] = []
    activities: list[np.ndarray] = []
    module.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            outputs = module.model(
                frames=batch["frames"].to(device),
                muscle=batch["muscle"].to(device),
                activity_ids=batch["activity_id"].to(device),
                mask_ratio=0.5,
            )
            state = outputs.fast_state if outputs.fast_state is not None else outputs.fused
            features.append(state.detach().cpu().numpy())
            muscles.append(batch["muscle"].detach().cpu().numpy())
            activity = batch["activity_id"].detach().cpu().numpy()
            activities.append(np.repeat(activity[:, None], state.shape[1], axis=1))
    return {
        "features": np.concatenate(features, axis=0),
        "muscles": np.concatenate(muscles, axis=0),
        "activities": np.concatenate(activities, axis=0),
    }


def run_delayed_activity_probe(feature_bank: dict[str, np.ndarray], delays: list[int]) -> list[dict[str, float]]:
    features = feature_bank["features"]
    activities = feature_bank["activities"]
    rows: list[dict[str, float]] = []
    for delay in delays:
        if delay >= features.shape[1]:
            continue
        probe_x = features[:, delay:, :].reshape(-1, features.shape[-1])
        probe_y = activities[:, :-delay].reshape(-1) if delay > 0 else activities.reshape(-1)
        accuracy = linear_accuracy(probe_x, probe_y)
        if accuracy is not None:
            rows.append({"delay": float(delay), "activity_accuracy": accuracy})
    return rows


def run_nback_probe(feature_bank: dict[str, np.ndarray], nback: list[int], clusters: int) -> list[dict[str, float]]:
    features = feature_bank["features"]
    muscles = feature_bank["muscles"]
    flat_muscle = muscles.reshape(-1, muscles.shape[-1])
    n_clusters = min(max(2, int(clusters)), max(2, flat_muscle.shape[0] // 2))
    labels = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit_predict(flat_muscle)
    labels = labels.reshape(muscles.shape[0], muscles.shape[1])
    rows: list[dict[str, float]] = []
    for n in nback:
        if n <= 0 or n >= features.shape[1]:
            continue
        probe_x = features[:, n:, :].reshape(-1, features.shape[-1])
        probe_y = (labels[:, n:] == labels[:, :-n]).astype(np.int64).reshape(-1)
        accuracy = linear_accuracy(probe_x, probe_y)
        if accuracy is not None:
            rows.append({"n": float(n), "accuracy": accuracy, "positive_rate": float(probe_y.mean())})
    return rows


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = load_config(args.config)
    for override in args.override:
        apply_override(config, override)
    module = load_lightning_module(args.checkpoint, args.config, args.override, device)
    dataset = build_dataset(config, args.dataset_root, args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_egomuscle,
    )
    mask_ratio = float(config["training"].get("mask_ratio", 0.5))
    rows = [
        evaluate_gap(module, loader, device, gap=gap, args=args, mask_ratio=mask_ratio, max_batches=args.max_probe_batches)
        for gap in args.gaps
    ]
    feature_bank = collect_memory_features(module, loader, device, max_batches=args.max_probe_batches)
    delayed_activity = run_delayed_activity_probe(feature_bank, args.nback)
    nback_rows = run_nback_probe(feature_bank, args.nback, args.motion_clusters)
    payload = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "mask_video": bool(args.mask_video),
        "mask_muscle": bool(args.mask_muscle),
        "gap_retention": rows,
        "delayed_activity": delayed_activity,
        "nback_motion_recurrence": nback_rows,
        "retention_tau": fit_retention_tau(rows),
        "retention_auc": trapezoid_auc([-row["mse"] for row in rows], [row["gap"] for row in rows]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
