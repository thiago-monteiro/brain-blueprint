from __future__ import annotations

import argparse
from pathlib import Path

from egomuscle.data.dataset import discover_records, load_video_frames_uint8
from egomuscle.training.train import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute sampled/resized uint8 frame caches for EgoMuscle splits.")
    parser.add_argument("--config", type=Path, default=Path("egomuscle/training/config.yaml"))
    parser.add_argument("--split", choices=["train", "val", "test"], nargs="+", default=["train", "val", "test"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    n_frames = int(config["data"].get("n_frames", 16))
    image_size = int(config["data"].get("image_size", 224))

    for split in args.split:
        split_cfg = config["data"][split]
        cache_dir = Path(split_cfg["frame_cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        records = discover_records(
            split_cfg["clip_dir"],
            muscle_dir=split_cfg.get("muscle_dir"),
            metadata_path=split_cfg.get("metadata_path"),
        )
        built = 0
        skipped = 0
        for record in records:
            cache_path = cache_dir / f"{record.video_path.stem}.npy"
            if cache_path.exists() and not args.overwrite:
                skipped += 1
                continue
            frames = load_video_frames_uint8(record.video_path, n_frames=n_frames, image_size=image_size)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            import numpy as np

            np.save(cache_path, frames.numpy())
            built += 1
        print(f"{split}: built={built} skipped={skipped} total={len(records)} cache_dir={cache_dir}")


if __name__ == "__main__":
    main()
