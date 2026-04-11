"""Launch the scaling sweep with full-matrix two-seed paper defaults (18 runs, not 27)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)

    import argparse
    import random

    parser = argparse.ArgumentParser()
    parser.add_argument("--random-seeds", action="store_true", help="Replace explicit SCALING_SEEDS with 30 random seeds")
    args, _ = parser.parse_known_args()

    env = os.environ.copy()
    env.setdefault("SCALING_MATRIX", "primary")
    env.setdefault("MANIFEST", "experiments/results/scaling_manifest_paper.jsonl")
    env.setdefault("SCALING_PROTOCOL_NAME", "paper_scaling_full_matrix_2seed_v1")
    env.setdefault("SCALING_COMPUTE_MODE", "token_parity")
    env.setdefault("SCALING_REF_VIDEO_MODEL_NAME", "OpenGVLab/VideoMAEv2-Large")
    env.setdefault("SCALING_REF_HIDDEN", "256")

    train_cache = root / "data/processed/full_cache/train"
    val_cache = root / "data/processed/full_cache/val"
    if train_cache.is_dir():
        env.setdefault("SCALING_TRAIN_FULL_CACHE", str(train_cache))
    if val_cache.is_dir():
        env.setdefault("SCALING_VAL_FULL_CACHE", str(val_cache))

    if args.random_seeds:
        seeds_str = " ".join(str(random.randint(0, 2**31 - 1)) for _ in range(30))
        env["SCALING_SEEDS"] = seeds_str

    subprocess.run(["python", "experiments/run_scaling_sweep.py"], env=env, check=True)


if __name__ == "__main__":
    main()
