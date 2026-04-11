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
from egomuscle.model.adapters import AdapterLinear
from egomuscle.training.smfe_losses import gaussian_nll, uncertainty_error_correlation
from egomuscle.training.train import apply_override, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate p-bit stochasticity and adapter quantization sweeps.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("egomuscle/training/config.yaml"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--pbit-modes", nargs="+", default=["stochastic_straight_through", "deterministic_sigmoid"])
    parser.add_argument("--quantization-levels", nargs="+", default=["fp16", "26", "16", "8", "4", "2"])
    parser.add_argument("--stochastic-samples", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/pbit_quantization_sweep.json"))
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


def set_pbit_mode(module, mode: str) -> int:
    count = 0
    for submodule in module.model.modules():
        if hasattr(submodule, "mode") and hasattr(submodule, "to_logits") and hasattr(submodule, "to_output"):
            submodule.mode = mode
            count += 1
    return count


def set_adapter_quantization(module, level: str) -> int:
    count = 0
    levels = None if level == "fp16" else int(level)
    mode = "none" if levels is None else "qat"
    for submodule in module.model.modules():
        if isinstance(submodule, AdapterLinear):
            submodule.quantization_levels = levels
            submodule.quantization_mode = mode
            count += 1
    return count


def summarize_stochastic_predictions(predictions: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    stacked = torch.stack(predictions, dim=0)
    return stacked.mean(dim=0), stacked.var(dim=0, unbiased=False)


def evaluate(module, loader: DataLoader, device: torch.device, *, samples: int, max_batches: int | None) -> dict[str, float]:
    mses: list[float] = []
    nlls: list[float] = []
    uncertainty_corrs: list[float] = []
    ensemble_corrs: list[float] = []
    entropy_values: list[float] = []
    module.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            frames = batch["frames"].to(device)
            muscle = batch["muscle"].to(device)
            activity_ids = batch["activity_id"].to(device)
            predictions: list[torch.Tensor] = []
            first_output = None
            for _ in range(max(samples, 1)):
                output = module.model(frames=frames, muscle=muscle, activity_ids=activity_ids, mask_ratio=0.5)
                first_output = output if first_output is None else first_output
                if output.pred is None or output.target is None:
                    raise ValueError("Sweep requires muscle prediction outputs.")
                predictions.append(output.pred.detach())
            assert first_output is not None and first_output.target is not None
            mean_pred, ensemble_var = summarize_stochastic_predictions(predictions)
            target = first_output.target
            mses.append(float(F.mse_loss(mean_pred, target).detach().cpu().item()))
            actual_error = (mean_pred - target).pow(2).reshape(-1)
            ensemble_flat = ensemble_var.reshape(-1)
            if ensemble_flat.numel() > 1 and float(ensemble_flat.norm().detach().cpu()) > 0:
                centered_var = ensemble_flat - ensemble_flat.mean()
                centered_err = actual_error - actual_error.mean()
                denom = centered_var.norm() * centered_err.norm()
                if float(denom.detach().cpu()) > 0:
                    ensemble_corrs.append(float((centered_var @ centered_err / denom).detach().cpu().item()))
            if first_output.pred_mu is not None and first_output.pred_log_var is not None:
                nlls.append(float(gaussian_nll(first_output.pred_mu, first_output.pred_log_var, target).detach().cpu().item()))
                uncertainty_corrs.append(
                    float(uncertainty_error_correlation(first_output.pred_mu, first_output.pred_log_var, target).detach().cpu().item())
                )
            if first_output.pbit_entropy is not None:
                entropy_values.append(float(first_output.pbit_entropy.mean().detach().cpu().item()))
    result = {"mse": float(np.mean(mses))}
    if nlls:
        result["nll"] = float(np.mean(nlls))
    if uncertainty_corrs:
        result["predicted_variance_error_corr"] = float(np.mean(uncertainty_corrs))
    if ensemble_corrs:
        result["ensemble_variance_error_corr"] = float(np.mean(ensemble_corrs))
    if entropy_values:
        result["pbit_entropy_bits"] = float(np.mean(entropy_values))
    return result


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

    rows: list[dict[str, Any]] = []
    for mode in args.pbit_modes:
        pbit_modules = set_pbit_mode(module, mode)
        for level in args.quantization_levels:
            adapter_modules = set_adapter_quantization(module, level)
            metrics = evaluate(module, loader, device, samples=args.stochastic_samples, max_batches=args.max_batches)
            row = {
                "pbit_mode": mode,
                "quantization_levels": level,
                "pbit_modules": pbit_modules,
                "adapter_modules": adapter_modules,
                **metrics,
            }
            rows.append(row)

    payload = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "stochastic_samples": args.stochastic_samples,
        "rows": rows,
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
