"""Download MinT from KIT RADAR (DOI 10.35097/VDPCEFSThBWlDPFL) and extract to data/raw/mint_extracted."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

RADAR_ARCHIVE_URL = "https://radar.kit.edu/radar-backend/archives/VDPCEFSThBWlDPFL/versions/1/content"
OUTER_NAME = "10.35097-VDPCEFSThBWlDPFL.tar"


def download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Reusing existing file {dest} ({dest.stat().st_size} bytes)")
        return
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with dest.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, disable=total <= 0
        ) as bar:
            for c in r.iter_content(chunk_size=chunk):
                if c:
                    f.write(c)
                    bar.update(len(c))


def find_inner_zst(search_root: Path) -> Path | None:
    for p in search_root.rglob("MinT.tar.zst"):
        return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path("data/raw/mint_radar_work"))
    parser.add_argument("--mint-root", type=Path, default=Path("data/raw/mint_extracted"))
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract.")
    args = parser.parse_args()

    if any(args.mint_root.rglob("muscle_activations.pkl")) and not args.force:
        print(f"MinT already extracted under {args.mint_root}")
        return

    outer_tar = args.work_dir / OUTER_NAME
    extract_outer = args.work_dir / "outer_extracted"
    if args.force:
        shutil.rmtree(args.work_dir, ignore_errors=True)
        shutil.rmtree(args.mint_root, ignore_errors=True)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    download(RADAR_ARCHIVE_URL, outer_tar)

    print(f"Extracting outer tar {outer_tar} -> {extract_outer} …")
    if extract_outer.exists():
        shutil.rmtree(extract_outer)
    extract_outer.mkdir(parents=True)
    with tarfile.open(outer_tar, "r") as tf:
        tf.extractall(extract_outer)

    inner = find_inner_zst(extract_outer)
    if inner is None:
        raise FileNotFoundError(f"MinT.tar.zst not found under {extract_outer} (outer layout changed?)")

    print(f"Extracting zstd archive {inner} -> {args.mint_root} …")
    args.mint_root.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["tar", "-I", "zstd", "-xf", str(inner), "-C", str(args.mint_root)],
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Need GNU tar with zstd support (install package `zstd`). On Debian/Ubuntu: sudo apt install zstd"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "tar -I zstd failed. Install zstd and ensure tar supports -I (GNU tar)."
        ) from exc

    if not next(args.mint_root.rglob("muscle_activations.pkl"), None):
        raise RuntimeError(f"No muscle_activations.pkl under {args.mint_root} after extract.")

    print(f"MinT ready at {args.mint_root}")


if __name__ == "__main__":
    main()
