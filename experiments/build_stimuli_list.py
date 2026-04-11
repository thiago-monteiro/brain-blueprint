from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic ordered stimuli list for RSA/layerwise benchmark runs."
    )
    parser.add_argument(
        "--stimuli-dir",
        type=Path,
        required=True,
        help="Root directory containing benchmark stimuli images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output text file (one relative path per line).",
    )
    parser.add_argument(
        "--ext",
        action="append",
        default=[".jpg", ".jpeg", ".png"],
        help="Image extension to include (repeatable). Default: .jpg .jpeg .png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stimuli_dir = args.stimuli_dir.resolve()
    if not stimuli_dir.exists() or not stimuli_dir.is_dir():
        raise FileNotFoundError(f"Stimuli directory does not exist: {stimuli_dir}")

    exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.ext}
    image_paths = [path for path in stimuli_dir.rglob("*") if path.is_file() and path.suffix.lower() in exts]
    image_paths = sorted(image_paths, key=lambda path: path.relative_to(stimuli_dir).as_posix())
    if not image_paths:
        raise FileNotFoundError(f"No stimuli images found under {stimuli_dir} for extensions: {sorted(exts)}")

    rel_lines = [path.relative_to(stimuli_dir).as_posix() for path in image_paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(rel_lines) + "\n")

    print(f"Wrote {len(rel_lines)} stimuli to {args.output}")


if __name__ == "__main__":
    main()
