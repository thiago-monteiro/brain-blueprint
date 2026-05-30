from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paired BABEL recentered and raw-video views.")
    parser.add_argument("recenter_root", nargs="?", type=Path, default=Path("data/processed_body"))
    parser.add_argument("exo_root", nargs="?", type=Path, default=Path("data/processed_exo_real"))
    parser.add_argument("--raw-cache", type=Path, default=Path("data/raw/babel_renders"))
    parser.add_argument("--body-cache", type=Path, default=Path("data/raw/babel_recentered"))
    parser.add_argument("--render-device", default="cuda")
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--render-workers", type=int, default=4)
    parser.add_argument("--recenter-detect-every", type=int, default=6)
    parser.add_argument("--recenter-crop-scale", type=float, default=1.8)
    parser.add_argument("--recenter-min-score", type=float, default=0.65)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--skip-quality-filter", action="store_true")
    args, extra_args = parser.parse_known_args()
    args.extra_args = extra_args
    return args


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        "-m",
        "egomuscle.data.build_min_t_dataset",
        "--video-source",
        "babel_recenter",
        "--output-root",
        str(args.recenter_root),
        "--paired-output-root",
        str(args.exo_root),
        "--video-cache",
        str(args.body_cache),
        "--paired-video-cache",
        str(args.raw_cache),
        "--render-device",
        args.render_device,
        "--render-size",
        str(args.render_size),
        "--render-workers",
        str(args.render_workers),
        "--recenter-detect-every",
        str(args.recenter_detect_every),
        "--recenter-crop-scale",
        str(args.recenter_crop_scale),
        "--recenter-min-score",
        str(args.recenter_min_score),
        "--workers",
        str(args.workers),
    ]
    if args.skip_quality_filter:
        command.append("--skip-quality-filter")
    command.extend(args.extra_args)
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
