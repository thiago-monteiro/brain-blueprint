"""Download VideoMAE-Base weights into models/videomae-base."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("models/videomae-base"),
        help="Directory to populate (merged with existing config files).",
    )
    parser.add_argument("--repo-id", default="MCG-NJU/videomae-base")
    args = parser.parse_args()

    args.local_dir.mkdir(parents=True, exist_ok=True)
    weight_files = list(args.local_dir.glob("*.safetensors")) + list(args.local_dir.glob("pytorch_model.bin"))
    if weight_files:
        print(f"Already present: {[p.name for p in weight_files]!s} under {args.local_dir}")
        return

    print(f"Downloading {args.repo_id} -> {args.local_dir} …")
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(args.local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print("Done.")


if __name__ == "__main__":
    main()
