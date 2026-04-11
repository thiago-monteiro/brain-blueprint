import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("baker")

def load_video_full(path: Path, image_size: int = 224) -> np.ndarray:
    vr = VideoReader(str(path), ctx=cpu(0), width=image_size, height=image_size)
    indices = np.arange(len(vr))
    frames = vr.get_batch(indices).asnumpy()
    return np.transpose(frames, (0, 3, 1, 2))

def is_valid_npy(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        m = np.load(path, mmap_mode="r")
        if m.size == 0:
            return False
        if m.ndim != 4 or m.shape[1] != 3:
            return False
        return True
    except Exception:
        return False

def bake_clip(item: dict, output_dir: Path, image_size: int) -> bool:
    clip_id = item["clip_id"]
    video_path = Path(item["clip_path"])
    target_path = output_dir / f"{clip_id}.npy"
    temp_path = target_path.with_suffix(".tmp")
    if is_valid_npy(target_path):
        return False
    try:
        frames = load_video_full(video_path, image_size=image_size)
        with temp_path.open("wb") as f:
            np.save(f, frames)
        temp_path.replace(target_path)
        del frames
        gc.collect()
        return True
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        logger.error(f"Failed to bake {clip_id}: {e}")
        return False

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_path) as f:
        metadata = json.load(f)

    logger.info(f"Baking {len(metadata)} clips to {output_dir}...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = []
        for r in tqdm(executor.map(lambda x: bake_clip(x, output_dir, args.image_size), metadata), total=len(metadata)):
            results.append(r)

    baked = sum(1 for r in results if r)
    logger.info(f"Done! Baked {baked} new clips. Total in cache: {len(metadata)}")

if __name__ == "__main__":
    main()
