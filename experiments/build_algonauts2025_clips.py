from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

TR_DURATION = 1.49
N_FRAMES_PER_TR = 16

H5_KEY_PATTERN = re.compile(r"ses-\d+_task-(?P<movie>[a-z]+)(?P<part>\d+)(?:_run-(?P<run>\d+))?")

MOVIE10_DIRS = {
    "bourne": "bourne",
    "wolf": "wolf",
    "life": "life",
    "figures": "figures",
}


def build_movie_mapping(h5_path: Path) -> dict[str, dict[str, Any]]:
    import h5py

    mapping: dict[str, dict[str, Any]] = {}
    with h5py.File(h5_path, "r") as handle:
        for key in handle.keys():
            match = H5_KEY_PATTERN.match(key)
            if not match:
                continue
            movie = match.group("movie")
            part = match.group("part")
            n_trs = handle[key].shape[0]
            movie_dir = MOVIE10_DIRS.get(movie)
            if movie_dir is None:
                continue
            run = match.group("run")
            task_name = f"{movie}{part}" if run is None else f"{movie}{part}_run{run}"
            mapping[key] = {
                "movie": movie,
                "part": part,
                "run": run,
                "task_name": task_name,
                "n_trs": n_trs,
                "movie_dir": movie_dir,
            }
    return mapping


def compute_frame_indices(
    n_trs: int,
    movie_fps: float,
    tr_duration: float = TR_DURATION,
    n_frames: int = N_FRAMES_PER_TR,
) -> list[list[int]]:
    tr_indices: list[list[int]] = []
    for tr_idx in range(n_trs):
        tr_start = tr_idx * tr_duration
        tr_end = tr_start + tr_duration
        frame_start = int(round(tr_start * movie_fps))
        frame_end = int(round(tr_end * movie_fps))
        if frame_end - frame_start < n_frames:
            frame_end = frame_start + n_frames
        if frame_end - frame_start > n_frames:
            mid = (frame_start + frame_end) / 2.0
            half = n_frames / 2.0
            frame_start = max(0, int(round(mid - half)))
            frame_end = frame_start + n_frames
        frames = list(range(frame_start, frame_end))
        tr_indices.append(frames)
    return tr_indices


def build_manifest_for_movie(
    stimuli_root: Path,
    mapping: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []

    for h5_key, info in mapping.items():
        movie = info["movie"]
        part = info["part"]
        n_trs = info["n_trs"]
        movie_dir_name = info["movie_dir"]

        mkv_path = stimuli_root / movie_dir_name / f"{movie_dir_name}{part}.mkv"
        if not mkv_path.exists():
            print(f"  WARNING: MKV not found: {mkv_path}")
            continue

        try:
            import decord
            vr = decord.VideoReader(str(mkv_path))
            movie_fps = float(vr.get_avg_fps())
            total_frames = len(vr)
        except Exception as exc:
            print(f"  WARNING: Could not probe {mkv_path}: {exc}")
            continue

        tr_indices = compute_frame_indices(n_trs, movie_fps)

        for tr_idx, frames in enumerate(tr_indices):
            if max(frames) >= total_frames:
                frames = [min(f, total_frames - 1) for f in frames]
            manifest.append({
                "h5_key": h5_key,
                "task_name": info["task_name"],
                "movie": movie,
                "part": part,
                "tr_index": tr_idx,
                "mkv_path": str(mkv_path),
                "frame_indices": frames,
                "n_frames": len(frames),
            })

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build manifest JSONL mapping fMRI TRs to video frame indices for Algonauts 2025."
    )
    parser.add_argument(
        "--h5-path",
        type=Path,
        default=Path("data/raw/algonauts2025/fmri/sub-01/func/sub-01_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5"),
    )
    parser.add_argument("--stimuli-root", type=Path, default=Path("data/raw/algonauts2025/stimuli/movies/movie10"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/algonauts2025_clip_manifest.jsonl"))
    parser.add_argument("--subject", type=str, default="sub-01")
    args = parser.parse_args()

    print("Building movie-to-h5 mapping...")
    mapping = build_movie_mapping(args.h5_path)
    print(f"Found {len(mapping)} movie segments")

    print("Computing TR frame indices...")
    manifest = build_manifest_for_movie(args.stimuli_root, mapping)
    print(f"Generated {len(manifest)} TR entries")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for entry in manifest:
            handle.write(json.dumps(entry) + "\n")
    print(f"Wrote manifest: {args.output}")

    movies = set(e["movie"] for e in manifest)
    parts = set((e["movie"], e["part"]) for e in manifest)
    print(f"Movies: {sorted(movies)}")
    print(f"Movie segments: {len(parts)}")
    print(f"Total TRs: {len(manifest)}")


if __name__ == "__main__":
    main()
