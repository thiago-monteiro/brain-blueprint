import os
import sys
import subprocess
import argparse
import random
from datetime import datetime
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)

    parser = argparse.ArgumentParser()
    parser.add_argument("--random-seeds", action="store_true", help="Run 30 random seeds instead of the default explicit seeds")
    args = parser.parse_args()

    stamp = os.environ.get("STAMP", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.environ["WANDB_INIT_TIMEOUT"] = os.environ.get("WANDB_INIT_TIMEOUT", "180")

    scaling_manifest = f"experiments/results/clean_runs/{stamp}/scaling_manifest_trainable_flop_parity.jsonl"
    ablation_root = f"experiments/results/clean_runs/{stamp}/ablations_epochs100"

    os.makedirs(f"experiments/results/clean_runs/{stamp}", exist_ok=True)
    os.makedirs("data/processed/full_cache/train", exist_ok=True)
    os.makedirs("data/processed/full_cache/val", exist_ok=True)

    print("==================================================================")
    print(" Bake Full Cache")
    print("==================================================================")
    subprocess.run(["python", "experiments/bake_full_cache.py", "--metadata", "data/processed/train/metadata.json", "--output-dir", "data/processed/full_cache/train", "--workers", "12"], check=True)
    subprocess.run(["python", "experiments/bake_full_cache.py", "--metadata", "data/processed/val/metadata.json", "--output-dir", "data/processed/full_cache/val", "--workers", "12"], check=True)

    print("\n==================================================================")
    print(" 1/2 Scaling Law")
    print(" Protocol: trainable_flop_parity, constant LR, min 400 steps")
    print("==================================================================")
    env = os.environ.copy()
    env.update({
        "SCALING_USE_WANDB": "1",
        "SCALING_WANDB_PROJECT": "egomuscle",
        "SCALING_WANDB_ENTITY": "oculusdev124",
        "SCALING_WANDB_MODE": "online",
        "SCALING_WANDB_GROUP": f"clean_scaling_{stamp}",
        "SCALING_PROTOCOL_NAME": "clean_scaling_trainable_flop_parity_v1",
        "SCALING_BACKBONES": "base_v2|OpenGVLab/VideoMAEv2-Base\nlarge_v2|OpenGVLab/VideoMAEv2-Large\nhuge_v2|OpenGVLab/VideoMAEv2-Huge",
        "SCALING_HIDDENS": "128 256 512",
        "SCALING_SEEDS": "0 1 2",
        "SCALING_VIDEO_TRAINABLE_STRATEGY": "last_n",
        "SCALING_VIDEO_TRAINABLE_LAYERS": "2",
        "SCALING_VIDEO_UNFREEZE_EMBEDDINGS": "0",
        "SCALING_MAX_EPOCHS_REF": "100",
        "SCALING_COMPUTE_MODE": "trainable_flop_parity",
        "SCALING_ALLOW_BUDGET_CLAMP": "0",
        "SCALING_STEP_MIN": "400",
        "SCALING_EPOCH_MIN": "0",
        "SCALING_EPOCH_MAX": "200",
        "SCALING_LR_MODE": "constant",
        "SCALING_LEARNING_RATE_BASE": "3e-4",
        "SCALING_WARMUP_RATIO": "0.1",
        "SCALING_MUSCLE_NOISE_STD": "0.01",
        "SCALING_TEMPORAL_SAMPLE_MODE": "random_stride",
        "SCALING_NUM_WORKERS": "12",
        "SCALING_COMPILE": "1",
        "SCALING_SKIP_EXISTING": "0",
        "SCALING_VIRTUAL_SAMPLING": "1",
        "SCALING_TRAIN_FULL_CACHE": "data/processed/full_cache/train",
        "SCALING_VAL_FULL_CACHE": "data/processed/full_cache/val",
        "EVAL_TWENTE": "1",
        "MANIFEST": scaling_manifest,
    })
    if args.random_seeds:
        seeds_str = " ".join(str(random.randint(0, 2**31 - 1)) for _ in range(30))
        env["SCALING_SEEDS"] = seeds_str
    subprocess.run(["python", "experiments/run_scaling_sweep.py"], env=env, check=True)

    print("\n==================================================================")
    print(" 2/2 Ablations")
    print(" Protocol: fixed 100 epochs, no early stopping, no max_steps")
    print("==================================================================")
    if args.random_seeds:
        seeds = [int(s) for s in env.get("SCALING_SEEDS", "").split() if s]
    else:
        seeds = [0, 1, 2]
    for seed in seeds:
        cmd = [
            "python", "experiments/run_ablations_csv.py",
            "--root", f"{ablation_root}/seed_{seed}",
            "--ego-root", "data/processed",
            "--keep-going",
            "--override", f"seed={seed}",
            "--override", "model.video_model_name=OpenGVLab/VideoMAEv2-Base",
            "--override", "model.video_trainable_strategy=last_n",
            "--override", "model.video_trainable_layers=2",
            "--override", "model.video_unfreeze_embeddings=false",
            "--override", "model.muscle_hidden_dim=256",
            "--override", "model.label_vocab_size=1024",
            "--override", "model.fusion_dropout=0.1",
            "--override", "model.pred_dropout=0.1",
            "--override", "data.temporal_sample_mode=random_stride",
            "--override", "data.num_workers=12",
            "--override", "data.train.full_cache_dir=data/processed/full_cache/train",
            "--override", "data.val.full_cache_dir=data/processed/full_cache/val",
            "--override", "data.train.muscle_noise_std=0.01",
            "--override", "training.compile=true",
            "--override", "training.max_steps=null",
            "--override", "training.max_epochs=100",
            "--override", "training.early_stopping_patience=null",
            "--override", "training.warmup_epochs=5",
            "--override", "logging.use_wandb=true",
            "--override", "logging.project=egomuscle",
            "--override", "logging.entity=oculusdev124",
            "--override", f"logging.group=clean_ablations_{stamp}",
            "--override", "logging.tags=[ablations,clean,epochs100,base_v2,last_n2,h256]",
            "--override", f"logging.notes=seed_{seed}",
            "--override", "logging.wandb_mode=online"
        ]
        subprocess.run(cmd, check=True)

    print("\nDone.")
    print(f"Scaling manifest: {scaling_manifest}")
    print(f"Ablations root: {ablation_root}")
    print("W&B groups:")
    print(f"  clean_scaling_{stamp}")
    print(f"  clean_ablations_{stamp}")

if __name__ == "__main__":
    main()
