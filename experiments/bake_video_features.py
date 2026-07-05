from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from egomuscle.data.dataset import discover_records, _sample_indices, load_video_frames_uint8
from egomuscle.model.video_encoder import FrozenVideoEncoder, _encode_video_tokens, pool_video_tokens


def is_valid_npy(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        m = np.load(path, mmap_mode="r")
        return m.size > 0 and m.ndim == 2
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute frozen VideoMAE backbone features.")
    parser.add_argument("--clip-dir", type=str, required=True)
    parser.add_argument("--muscle-dir", type=str, default=None)
    parser.add_argument("--metadata-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--video-model-name", type=str, default="MCG-NJU/videomae-base")
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = discover_records(args.clip_dir, args.muscle_dir, args.metadata_path)
    print(f"Found {len(records)} records", flush=True)

    encoder = FrozenVideoEncoder(args.video_model_name).to(args.device)
    encoder.eval()
    for param in encoder.encoder.parameters():
        param.requires_grad = False
    hidden_dim = encoder.hidden_dim
    patch_size = encoder.patch_size
    tubelet_size = encoder.tubelet_size
    image_size = getattr(encoder.encoder.config, "image_size", args.image_size)

    baked = 0
    skipped = 0

    for record in tqdm(records):
        stem = record.video_path.stem
        target_path = output_dir / f"{stem}.npy"
        if is_valid_npy(target_path):
            skipped += 1
            continue

        total_frames = record.frame_count
        if total_frames is None or total_frames <= 0:
            try:
                from egomuscle.data.dataset import get_video_frame_count
                total_frames = get_video_frame_count(record.video_path)
            except Exception:
                print(f"  skip {stem}: cannot determine frame count", flush=True)
                continue

        indices = _sample_indices(total_frames, args.n_frames, mode="sparse_uniform")
        frames = load_video_frames_uint8(record.video_path, n_frames=args.n_frames, image_size=args.image_size, indices=indices)
        frames = frames.unsqueeze(0).to(args.device)

        with torch.no_grad():
            with torch.autocast(device_type=args.device, enabled=args.device == "cuda"):
                outputs = _encode_video_tokens(encoder.encoder, frames, return_layerwise=False)
                pooled = pool_video_tokens(
                    outputs.last_hidden_state,
                    target_frames=frames.shape[1],
                    image_size=image_size,
                    patch_size=patch_size,
                    tubelet_size=tubelet_size,
                )

        features = pooled.squeeze(0).float().cpu().numpy()
        np.save(target_path, features)
        baked += 1

        del frames, outputs, pooled
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print(f"Done: baked {baked} new, skipped {skipped} existing -> {output_dir}", flush=True)


if __name__ == "__main__":
    main()
