from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Condition:
    key: str
    run_name: str
    overrides: tuple[str, ...]


SMFE_CONDITIONS: tuple[Condition, ...] = (
    Condition(
        key="B0",
        run_name="B0_E3_current",
        overrides=(
            "training.loss_name=current",
            "training.loss_mode=mse",
            "model.use_video=true",
            "model.use_muscle=true",
            "model.fusion_mode=cross_attn",
            "model.predictive_distribution=point",
            "model.video_latent_prediction=false",
            "model.fast_memory.enabled=false",
            "model.pbit.enabled=false",
            "model.slow_adapter.enabled=false",
        ),
    ),
    Condition(
        key="B1",
        run_name="B1_E4_exocentric_current",
        overrides=(
            "training.loss_name=current",
            "training.loss_mode=mse",
            "model.use_video=true",
            "model.use_muscle=true",
            "model.fusion_mode=cross_attn",
            "data.train.clip_dir=data/processed_exo/train/clips",
            "data.val.clip_dir=data/processed_exo/val/clips",
            "data.test.clip_dir=data/processed_exo/test/clips",
            "data.train.metadata_path=data/processed_exo/train/metadata.json",
            "data.val.metadata_path=data/processed_exo/val/metadata.json",
            "data.test.metadata_path=data/processed_exo/test/metadata.json",
            "data.train.frame_cache_dir=data/processed_exo/train/frames",
            "data.val.frame_cache_dir=data/processed_exo/val/frames",
            "data.test.frame_cache_dir=data/processed_exo/test/frames",
        ),
    ),
    Condition(
        key="B2",
        run_name="B2_E5_scrambled_current",
        overrides=(
            "training.loss_name=current",
            "training.loss_mode=mse",
            "model.use_video=true",
            "model.use_muscle=true",
            "model.fusion_mode=cross_attn",
            "data.train.scramble_video=true",
            "data.val.scramble_video=true",
            "data.test.scramble_video=true",
        ),
    ),
    Condition(
        key="S0",
        run_name="S0_smfe_no_memory",
        overrides=(
            "training.loss_name=smfe",
            "model.predictive_distribution=gaussian",
            "model.video_latent_prediction=true",
            "model.fast_memory.enabled=false",
            "model.pbit.enabled=false",
            "model.slow_adapter.enabled=false",
            "training.loss_weights.muscle_nll=1.0",
            "training.loss_weights.video_latent=0.1",
            "training.loss_weights.precision=0.05",
        ),
    ),
    Condition(
        key="S1",
        run_name="S1_smfe_fast_memory",
        overrides=(
            "training.loss_name=smfe",
            "model.predictive_distribution=gaussian",
            "model.video_latent_prediction=true",
            "model.fast_memory.enabled=true",
            "model.pbit.enabled=false",
            "model.slow_adapter.enabled=false",
            "training.loss_weights.muscle_nll=1.0",
            "training.loss_weights.video_latent=0.1",
            "training.loss_weights.fast_kl=0.01",
            "training.loss_weights.precision=0.05",
        ),
    ),
    Condition(
        key="S2",
        run_name="S2_smfe_fast_memory_pbit",
        overrides=(
            "training.loss_name=smfe",
            "model.predictive_distribution=gaussian",
            "model.video_latent_prediction=true",
            "model.fast_memory.enabled=true",
            "model.pbit.enabled=true",
            "model.pbit.mode=stochastic_straight_through",
            "model.slow_adapter.enabled=false",
            "training.loss_weights.entropy=0.01",
            "training.loss_weights.homeostasis=0.01",
        ),
    ),
    Condition(
        key="S4",
        run_name="S4_full_smfe_memory",
        overrides=(
            "training.loss_name=smfe",
            "model.predictive_distribution=gaussian",
            "model.video_latent_prediction=true",
            "model.fast_memory.enabled=true",
            "model.pbit.enabled=true",
            "model.pbit.mode=stochastic_straight_through",
            "model.slow_adapter.enabled=true",
            "model.slow_adapter.quantization_levels=26",
            "model.slow_adapter.quantization_mode=qat",
            "training.precision=32-true",
            "training.loss_weights.entropy=0.01",
            "training.loss_weights.homeostasis=0.01",
            "training.loss_weights.capacity=0.001",
            "training.loss_weights.plastic=0.001",
        ),
    ),
    Condition(
        key="S3",
        run_name="S3_smfe_fast_memory_slow_adapter",
        overrides=(
            "training.loss_name=smfe",
            "model.predictive_distribution=gaussian",
            "model.video_latent_prediction=true",
            "model.fast_memory.enabled=true",
            "model.pbit.enabled=false",
            "model.slow_adapter.enabled=true",
            "model.slow_adapter.quantization_levels=null",
            "model.slow_adapter.quantization_mode=none",
            "training.loss_weights.capacity=0.0",
            "training.loss_weights.plastic=0.001",
        ),
    ),
    Condition(
        key="S5",
        run_name="S5_full_deterministic_control",
        overrides=(
            "training.loss_name=smfe",
            "model.predictive_distribution=gaussian",
            "model.video_latent_prediction=true",
            "model.fast_memory.enabled=true",
            "model.pbit.enabled=true",
            "model.pbit.mode=deterministic_sigmoid",
            "model.slow_adapter.enabled=true",
            "model.slow_adapter.quantization_levels=26",
            "model.slow_adapter.quantization_mode=qat",
            "training.precision=32-true",
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch SMFE-Memory ablation conditions.")
    parser.add_argument("--config", type=Path, default=Path("egomuscle/training/config.yaml"))
    parser.add_argument("--only", nargs="+", default=["S0", "S1"], help="Condition keys to run.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [condition for condition in SMFE_CONDITIONS if condition.key in set(args.only)]
    if not selected:
        raise ValueError(f"No conditions selected from: {', '.join(c.key for c in SMFE_CONDITIONS)}")

    for condition in selected:
        command = [
            sys.executable,
            "-m",
            "egomuscle.training.train",
            "--config",
            str(args.config),
            "--override",
            f"logging.run_name={condition.run_name}",
        ]
        for override in (*condition.overrides, *args.override):
            command.extend(["--override", override])
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
