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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from egomuscle.data.dataset import EgoMuscleDataset, collate_egomuscle
from egomuscle.eval.twente_eval import load_lightning_module
from egomuscle.training.smfe_losses import gaussian_nll
from egomuscle.training.train import apply_override, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agency-mismatch and self-boundary proxy probes.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("egomuscle/training/config.yaml"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path, default=Path("experiments/results/agency_boundary_probe.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int)
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


def prediction_metrics(output) -> dict[str, float]:
    if output.pred is None or output.target is None:
        raise ValueError("Agency probe requires muscle predictions.")
    metrics = {"mse": float(F.mse_loss(output.pred, output.target).detach().cpu().item())}
    if output.pred_mu is not None and output.pred_log_var is not None:
        metrics["nll"] = float(gaussian_nll(output.pred_mu, output.pred_log_var, output.target).detach().cpu().item())
        metrics["predicted_variance"] = float(torch.exp(output.pred_log_var).mean().detach().cpu().item())
    if output.fast_state is not None:
        metrics["fast_state_norm"] = float(output.fast_state.norm(dim=-1).mean().detach().cpu().item())
    if output.pbit_entropy is not None:
        metrics["pbit_entropy_bits"] = float(output.pbit_entropy.mean().detach().cpu().item())
    return metrics


def mismatch_batch(batch: dict[str, Any]) -> dict[str, Any]:
    mixed = dict(batch)
    batch_size = batch["frames"].shape[0]
    if batch_size < 2:
        return mixed
    permutation = torch.roll(torch.arange(batch_size), shifts=1)
    mixed["muscle"] = batch["muscle"][permutation]
    mixed["activity_id"] = batch["activity_id"][permutation]
    return mixed


def evaluate(module, loader: DataLoader, device: torch.device, *, max_batches: int | None) -> dict[str, Any]:
    matched_rows: list[dict[str, float]] = []
    mismatched_rows: list[dict[str, float]] = []
    delta_state: list[float] = []
    module.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if batch["frames"].shape[0] < 2:
                continue
            frames = batch["frames"].to(device)
            muscle = batch["muscle"].to(device)
            activity_ids = batch["activity_id"].to(device)
            matched = module.model(frames=frames, muscle=muscle, activity_ids=activity_ids, mask_ratio=0.5)
            mixed = mismatch_batch({"frames": frames, "muscle": muscle, "activity_id": activity_ids})
            mismatched = module.model(
                frames=mixed["frames"],
                muscle=mixed["muscle"],
                activity_ids=mixed["activity_id"],
                mask_ratio=0.5,
            )
            matched_rows.append(prediction_metrics(matched))
            mismatched_rows.append(prediction_metrics(mismatched))
            if matched.fast_state is not None and mismatched.fast_state is not None:
                delta_state.append(float((matched.fast_state - mismatched.fast_state).norm(dim=-1).mean().detach().cpu().item()))

    def average(rows: list[dict[str, float]]) -> dict[str, float]:
        keys = sorted({key for row in rows for key in row})
        return {key: float(np.mean([row[key] for row in rows if key in row])) for key in keys}

    matched_avg = average(matched_rows)
    mismatched_avg = average(mismatched_rows)
    delta = {f"delta_{key}": mismatched_avg[key] - matched_avg[key] for key in matched_avg.keys() & mismatched_avg.keys()}
    if delta_state:
        delta["delta_fast_state_distance"] = float(np.mean(delta_state))
    return {"matched": matched_avg, "mismatched": mismatched_avg, "delta": delta}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = load_config(args.config)
    for override in args.override:
        apply_override(config, override)
    module = load_lightning_module(args.checkpoint, args.config, args.override, device)
    dataset = build_dataset(config, args.dataset_root, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_egomuscle)
    result = evaluate(module, loader, device, max_batches=args.max_batches)
    payload = {"checkpoint": str(args.checkpoint), "split": args.split, **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        row = {f"{group}_{key}": value for group in ("matched", "mismatched", "delta") for key, value in result[group].items()}
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
