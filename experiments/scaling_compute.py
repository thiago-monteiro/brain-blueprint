from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from torch.utils.flop_counter import FlopCounterMode
from transformers import VideoMAEConfig

from egomuscle.data.dataset import discover_records
from egomuscle.model.egomuscle import EgoMuscleModel


CACHE_PATH = Path("experiments/results/scaling_compute_cache.json")


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_cache(path: Path = CACHE_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_cache(cache: dict[str, Any], path: Path = CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def _device_count(devices: Any) -> int:
    if devices in (None, "auto"):
        return 1
    if isinstance(devices, int):
        return max(1, devices)
    if isinstance(devices, (list, tuple)):
        return max(1, len(devices))
    return 1


def _video_seq_len(cfg: VideoMAEConfig, n_frames: int, image_size: int) -> int:
    patch = int(getattr(cfg, "patch_size", 16))
    tubelet = int(getattr(cfg, "tubelet_size", 2))
    spatial = max(1, image_size // patch) ** 2
    temporal = max(1, n_frames // tubelet)
    cls_tokens = 1
    return spatial * temporal + cls_tokens


def _training_sample_count(config: dict[str, Any]) -> int:
    train_cfg = config.get("data", {}).get("train")
    if not train_cfg:
        raise ValueError("Config does not define data.train")
    common = config.get("data", {})
    records = discover_records(
        video_dir=train_cfg["clip_dir"],
        muscle_dir=train_cfg.get("muscle_dir"),
        metadata_path=train_cfg.get("metadata_path"),
    )
    return len(records)


def _micro_batches_per_epoch(train_samples: int, batch_size: int, devices: int) -> int:
    per_step_samples = max(1, int(batch_size) * int(devices))
    return max(1, math.ceil(train_samples / per_step_samples))


def _optimizer_steps_per_epoch(train_samples: int, batch_size: int, accumulate_grad_batches: int, devices: int) -> int:
    micro_batches = _micro_batches_per_epoch(train_samples, batch_size=batch_size, devices=devices)
    return max(1, math.ceil(micro_batches / max(1, int(accumulate_grad_batches))))


@dataclass
class ForwardFlopEstimate:
    total_forward_flops_per_sample: float
    frozen_forward_flops_per_sample: float
    trainable_forward_flops_per_sample: float


@dataclass
class CostEstimate:
    train_samples: int
    device_count: int
    micro_batches_per_epoch: int
    optimizer_steps_per_epoch: int
    video_tokens_per_sample: int
    effective_batch_size: int
    video_tokens_per_optimizer_step: int
    total_forward_flops_per_sample: float
    frozen_forward_flops_per_sample: float
    trainable_forward_flops_per_sample: float
    train_flops_per_optimizer_step: float


def _flop_cache_key(
    *,
    video_model_name: str,
    n_frames: int,
    image_size: int,
    muscle_dim: int,
    prediction_dim: int,
    muscle_hidden_dim: int,
    fusion_mode: str,
    use_video: bool,
    use_muscle: bool,
    label_conditioning: bool,
    video_trainable_strategy: str,
    video_trainable_layers: int,
    video_unfreeze_embeddings: bool,
) -> str:
    payload = {
        "video_model_name": video_model_name,
        "n_frames": int(n_frames),
        "image_size": int(image_size),
        "muscle_dim": int(muscle_dim),
        "prediction_dim": int(prediction_dim),
        "muscle_hidden_dim": int(muscle_hidden_dim),
        "fusion_mode": fusion_mode,
        "use_video": bool(use_video),
        "use_muscle": bool(use_muscle),
        "label_conditioning": bool(label_conditioning),
        "video_trainable_strategy": video_trainable_strategy,
        "video_trainable_layers": int(video_trainable_layers),
        "video_unfreeze_embeddings": bool(video_unfreeze_embeddings),
    }
    return json.dumps(payload, sort_keys=True)


def _measure_forward_flops(
    *,
    video_model_name: str,
    n_frames: int,
    image_size: int,
    muscle_dim: int,
    prediction_dim: int,
    muscle_hidden_dim: int,
    fusion_mode: str,
    use_video: bool,
    use_muscle: bool,
    label_conditioning: bool,
    video_trainable_strategy: str,
    video_trainable_layers: int,
    video_unfreeze_embeddings: bool,
) -> ForwardFlopEstimate:
    key = _flop_cache_key(
        video_model_name=video_model_name,
        n_frames=n_frames,
        image_size=image_size,
        muscle_dim=muscle_dim,
        prediction_dim=prediction_dim,
        muscle_hidden_dim=muscle_hidden_dim,
        fusion_mode=fusion_mode,
        use_video=use_video,
        use_muscle=use_muscle,
        label_conditioning=label_conditioning,
        video_trainable_strategy=video_trainable_strategy,
        video_trainable_layers=video_trainable_layers,
        video_unfreeze_embeddings=video_unfreeze_embeddings,
    )
    cache = _load_cache()
    if key in cache:
        return ForwardFlopEstimate(**cache[key])

    model = EgoMuscleModel(
        video_model_name=video_model_name,
        muscle_dim=muscle_dim,
        prediction_dim=prediction_dim,
        muscle_hidden_dim=muscle_hidden_dim,
        fusion_mode=fusion_mode,
        use_video=use_video,
        use_muscle=use_muscle,
        label_conditioning=label_conditioning,
        video_trainable_strategy=video_trainable_strategy,
        video_trainable_layers=video_trainable_layers,
        video_unfreeze_embeddings=video_unfreeze_embeddings,
    ).cpu()
    model.eval()

    frames = torch.randn(1, n_frames, 3, image_size, image_size, dtype=torch.float32) if use_video else None
    muscle = torch.randn(1, n_frames, muscle_dim, dtype=torch.float32) if use_muscle else None
    activity_ids = torch.zeros(1, dtype=torch.long) if label_conditioning else None

    with torch.no_grad():
        with FlopCounterMode(display=False) as flop_counter:
            model(frames=frames, muscle=muscle, activity_ids=activity_ids, mask_ratio=0.5)
    counts = flop_counter.get_flop_counts()
    total = float(sum(counts["EgoMuscleModel"].values()))
    frozen = float(sum(counts.get("EgoMuscleModel.video_encoder", {}).values()))
    estimate = ForwardFlopEstimate(
        total_forward_flops_per_sample=total,
        frozen_forward_flops_per_sample=frozen,
        trainable_forward_flops_per_sample=max(0.0, total - frozen),
    )
    cache[key] = asdict(estimate)
    _write_cache(cache)
    return estimate


def estimate_cost(
    *,
    config: dict[str, Any],
    video_model_name: str,
    muscle_hidden_dim: int,
    batch_size: int,
    accumulate_grad_batches: int,
    fusion_mode: str,
    prediction_dim: int,
    use_video: bool,
    use_muscle: bool,
    label_conditioning: bool,
    video_trainable_strategy: str,
    video_trainable_layers: int,
    video_unfreeze_embeddings: bool,
) -> CostEstimate:
    data_cfg = config.get("data", {})
    trainer_cfg = config.get("trainer", {})
    n_frames = int(data_cfg.get("n_frames", 16))
    image_size = int(data_cfg.get("image_size", 224))
    muscle_dim = int(data_cfg.get("muscle_dim", 32))
    devices = _device_count(trainer_cfg.get("devices", "auto"))
    train_samples = _training_sample_count(config)
    micro_batches = _micro_batches_per_epoch(train_samples, batch_size=batch_size, devices=devices)
    optimizer_steps = _optimizer_steps_per_epoch(
        train_samples,
        batch_size=batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
        devices=devices,
    )

    video_cfg = VideoMAEConfig.from_pretrained(video_model_name, trust_remote_code=True)
    video_tokens_per_sample = _video_seq_len(video_cfg, n_frames=n_frames, image_size=image_size)
    effective_batch_size = int(batch_size) * int(accumulate_grad_batches) * int(devices)

    forward = _measure_forward_flops(
        video_model_name=video_model_name,
        n_frames=n_frames,
        image_size=image_size,
        muscle_dim=muscle_dim,
        prediction_dim=prediction_dim,
        muscle_hidden_dim=muscle_hidden_dim,
        fusion_mode=fusion_mode,
        use_video=use_video,
        use_muscle=use_muscle,
        label_conditioning=label_conditioning,
        video_trainable_strategy=video_trainable_strategy,
        video_trainable_layers=video_trainable_layers,
        video_unfreeze_embeddings=video_unfreeze_embeddings,
    )
    train_flops_per_sample = forward.frozen_forward_flops_per_sample + (3.0 * forward.trainable_forward_flops_per_sample)
    return CostEstimate(
        train_samples=train_samples,
        device_count=devices,
        micro_batches_per_epoch=micro_batches,
        optimizer_steps_per_epoch=optimizer_steps,
        video_tokens_per_sample=int(video_tokens_per_sample),
        effective_batch_size=effective_batch_size,
        video_tokens_per_optimizer_step=int(video_tokens_per_sample) * effective_batch_size,
        total_forward_flops_per_sample=forward.total_forward_flops_per_sample,
        frozen_forward_flops_per_sample=forward.frozen_forward_flops_per_sample,
        trainable_forward_flops_per_sample=forward.trainable_forward_flops_per_sample,
        train_flops_per_optimizer_step=float(train_flops_per_sample * effective_batch_size),
    )


def _reference_budgets(
    *,
    baseline_epochs: int,
    reference_cost: CostEstimate,
) -> tuple[int, float]:
    ref_steps = int(baseline_epochs) * int(reference_cost.optimizer_steps_per_epoch)
    ref_tokens = ref_steps * int(reference_cost.video_tokens_per_optimizer_step)
    ref_flops = float(ref_steps) * float(reference_cost.train_flops_per_optimizer_step)
    return ref_tokens, ref_flops


def schedule_training_budget(
    *,
    mode: str,
    baseline_epochs: int,
    target_cost: CostEstimate,
    reference_cost: CostEstimate,
    step_min: int,
    step_max: int,
) -> dict[str, Any]:
    reference_total_tokens, reference_total_flops = _reference_budgets(
        baseline_epochs=baseline_epochs,
        reference_cost=reference_cost,
    )
    unclamped_max_steps: float
    if mode == "fixed":
        unclamped_max_steps = float(baseline_epochs * target_cost.optimizer_steps_per_epoch)
    elif mode == "token_parity":
        unclamped_max_steps = float(reference_total_tokens) / max(1, target_cost.video_tokens_per_optimizer_step)
    elif mode == "flop_parity":
        unclamped_max_steps = float(reference_total_flops) / max(1.0, target_cost.train_flops_per_optimizer_step)
    elif mode == "trainable_flop_parity":
        ref_steps = int(baseline_epochs) * int(reference_cost.optimizer_steps_per_epoch)
        ref_frozen_total = float(ref_steps) * float(reference_cost.frozen_forward_flops_per_sample) * float(reference_cost.effective_batch_size)
        ref_trainable_total = reference_total_flops - ref_frozen_total
        target_frozen_per_step = float(target_cost.frozen_forward_flops_per_sample) * float(target_cost.effective_batch_size)
        target_trainable_per_step = max(1.0, target_cost.train_flops_per_optimizer_step - target_frozen_per_step)
        unclamped_max_steps = float(ref_trainable_total) / target_trainable_per_step
    else:
        raise ValueError(f"Unsupported parity mode: {mode}")

    rounded_max_steps = int(round(unclamped_max_steps))
    max_steps = max(step_min, min(step_max, rounded_max_steps))
    max_epochs = max(1, math.ceil(max_steps / max(1, target_cost.optimizer_steps_per_epoch)))
    clamp_reason = None
    if max_steps != rounded_max_steps:
        if max_steps == step_min:
            clamp_reason = "step_min"
        elif max_steps == step_max:
            clamp_reason = "step_max"
        else:
            clamp_reason = "other"
    return {
        "mode": mode,
        "max_steps": int(max_steps),
        "max_epochs_cap": int(max_epochs),
        "unclamped_max_steps": float(unclamped_max_steps),
        "rounded_max_steps": int(rounded_max_steps),
        "step_min": int(step_min),
        "step_max": int(step_max),
        "budget_was_clamped": bool(clamp_reason is not None),
        "budget_clamp_reason": clamp_reason,
        "reference_total_video_tokens": int(reference_total_tokens),
        "reference_total_train_flops": float(reference_total_flops),
    }


def _shell_lines(payload: dict[str, object]) -> str:
    lines: list[str] = []
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, float):
            lines.append(f"{key}={value:.6f}")
        else:
            lines.append(f"{key}={value}")
    return "\n".join(lines)


def cmd_schedule(args: argparse.Namespace) -> None:
    config = _load_yaml(args.config)
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})

    baseline_epochs = int(args.baseline_epochs or training_cfg.get("max_epochs", 100))
    fusion_mode = str(args.fusion_mode or model_cfg.get("fusion_mode", "cross_attn"))
    prediction_dim = int(args.prediction_dim or model_cfg.get("prediction_dim", data_cfg.get("muscle_dim", 32)))
    use_video = bool(args.use_video if args.use_video is not None else model_cfg.get("use_video", True))
    use_muscle = bool(args.use_muscle if args.use_muscle is not None else model_cfg.get("use_muscle", True))
    label_conditioning = bool(
        args.label_conditioning if args.label_conditioning is not None else model_cfg.get("label_conditioning", False)
    )
    video_trainable_strategy = str(args.video_trainable_strategy or model_cfg.get("video_trainable_strategy", "frozen"))
    video_trainable_layers = int(args.video_trainable_layers if args.video_trainable_layers is not None else model_cfg.get("video_trainable_layers", 0))
    video_unfreeze_embeddings = bool(
        args.video_unfreeze_embeddings
        if args.video_unfreeze_embeddings is not None
        else model_cfg.get("video_unfreeze_embeddings", False)
    )

    ref_video = args.ref_video_model_name or str(model_cfg.get("video_model_name", "MCG-NJU/videomae-base"))
    ref_hidden = int(args.ref_muscle_hidden_dim or model_cfg.get("muscle_hidden_dim", 128))
    ref_batch = int(args.ref_batch_size or data_cfg.get("batch_size", 32))
    ref_accum = int(args.ref_accumulate_grad_batches or training_cfg.get("accumulate_grad_batches", 4))

    target_cost = estimate_cost(
        config=config,
        video_model_name=args.video_model_name,
        muscle_hidden_dim=int(args.muscle_hidden_dim),
        batch_size=int(args.batch_size),
        accumulate_grad_batches=int(args.accumulate_grad_batches),
        fusion_mode=fusion_mode,
        prediction_dim=prediction_dim,
        use_video=use_video,
        use_muscle=use_muscle,
        label_conditioning=label_conditioning,
        video_trainable_strategy=video_trainable_strategy,
        video_trainable_layers=video_trainable_layers,
        video_unfreeze_embeddings=video_unfreeze_embeddings,
    )
    reference_cost = estimate_cost(
        config=config,
        video_model_name=ref_video,
        muscle_hidden_dim=ref_hidden,
        batch_size=ref_batch,
        accumulate_grad_batches=ref_accum,
        fusion_mode=fusion_mode,
        prediction_dim=prediction_dim,
        use_video=use_video,
        use_muscle=use_muscle,
        label_conditioning=label_conditioning,
        video_trainable_strategy=video_trainable_strategy,
        video_trainable_layers=video_trainable_layers,
        video_unfreeze_embeddings=video_unfreeze_embeddings,
    )

    step_min = int(args.step_min or int(args.epoch_min) * target_cost.optimizer_steps_per_epoch)
    step_max = int(args.step_max or int(args.epoch_max) * target_cost.optimizer_steps_per_epoch)
    schedule = schedule_training_budget(
        mode=args.mode,
        baseline_epochs=baseline_epochs,
        target_cost=target_cost,
        reference_cost=reference_cost,
        step_min=step_min,
        step_max=step_max,
    )
    payload = {
        **schedule,
        "train_samples": target_cost.train_samples,
        "device_count": target_cost.device_count,
        "micro_batches_per_epoch": target_cost.micro_batches_per_epoch,
        "optimizer_steps_per_epoch": target_cost.optimizer_steps_per_epoch,
        "video_tokens_per_sample": target_cost.video_tokens_per_sample,
        "effective_batch_size": target_cost.effective_batch_size,
        "video_tokens_per_optimizer_step": target_cost.video_tokens_per_optimizer_step,
        "total_forward_flops_per_sample": target_cost.total_forward_flops_per_sample,
        "frozen_forward_flops_per_sample": target_cost.frozen_forward_flops_per_sample,
        "trainable_forward_flops_per_sample": target_cost.trainable_forward_flops_per_sample,
        "train_flops_per_optimizer_step": target_cost.train_flops_per_optimizer_step,
        "ref_video_model_name": ref_video,
        "ref_muscle_hidden_dim": ref_hidden,
        "ref_effective_batch_size": reference_cost.effective_batch_size,
        "ref_optimizer_steps_per_epoch": reference_cost.optimizer_steps_per_epoch,
        "ref_video_tokens_per_optimizer_step": reference_cost.video_tokens_per_optimizer_step,
        "ref_train_flops_per_optimizer_step": reference_cost.train_flops_per_optimizer_step,
        "baseline_epochs": baseline_epochs,
        "video_trainable_strategy": video_trainable_strategy,
        "video_trainable_layers": video_trainable_layers,
        "video_unfreeze_embeddings": int(video_unfreeze_embeddings),
    }
    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(_shell_lines(payload))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate scaling-law compute budgets.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sched = sub.add_parser("schedule")
    sched.add_argument("--config", type=Path, required=True)
    sched.add_argument("--video-model-name", required=True)
    sched.add_argument("--muscle-hidden-dim", type=int, required=True)
    sched.add_argument("--batch-size", type=int, required=True)
    sched.add_argument("--accumulate-grad-batches", type=int, required=True)
    sched.add_argument("--mode", choices=("fixed", "token_parity", "flop_parity", "trainable_flop_parity"), default="flop_parity")
    sched.add_argument("--baseline-epochs", type=int, default=None)
    sched.add_argument("--epoch-min", type=int, default=4)
    sched.add_argument("--epoch-max", type=int, default=200)
    sched.add_argument("--step-min", type=int, default=None)
    sched.add_argument("--step-max", type=int, default=None)
    sched.add_argument("--ref-video-model-name", default=None)
    sched.add_argument("--ref-muscle-hidden-dim", type=int, default=None)
    sched.add_argument("--ref-batch-size", type=int, default=None)
    sched.add_argument("--ref-accumulate-grad-batches", type=int, default=None)
    sched.add_argument("--prediction-dim", type=int, default=None)
    sched.add_argument("--fusion-mode", default=None)
    sched.add_argument("--use-video", type=int, choices=(0, 1), default=None)
    sched.add_argument("--use-muscle", type=int, choices=(0, 1), default=None)
    sched.add_argument("--label-conditioning", type=int, choices=(0, 1), default=None)
    sched.add_argument("--video-trainable-strategy", choices=("frozen", "adapter_only", "last_n", "full"), default=None)
    sched.add_argument("--video-trainable-layers", type=int, default=None)
    sched.add_argument("--video-unfreeze-embeddings", type=int, choices=(0, 1), default=None)
    sched.add_argument("--format", choices=("json", "shell"), default="json")
    sched.set_defaults(func=cmd_schedule)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
