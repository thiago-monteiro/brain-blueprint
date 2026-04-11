from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _count_suffixes(root: Path, suffixes: tuple[str, ...]) -> int:
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def summarize_amass(amass_root: Path) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for dataset_dir in sorted(path for path in amass_root.iterdir() if path.is_dir()):
        summary[dataset_dir.name] = {
            "npz_files": _count_suffixes(dataset_dir, (".npz",)),
            "directories": sum(1 for path in dataset_dir.rglob("*") if path.is_dir()),
        }
    return summary


def summarize_twente(rgb_root: Path, emg_root: Path) -> dict[str, object]:
    videos_root = rgb_root / "Videos_anonymized"
    processed_root = emg_root / "Processed_data"
    video_subjects = sorted(path.name for path in videos_root.iterdir() if path.is_dir()) if videos_root.exists() else []
    emg_subjects = sorted(path.name for path in processed_root.iterdir() if path.is_dir() and path.name.startswith("Subj"))
    return {
        "video_subjects": video_subjects,
        "emg_subjects": emg_subjects,
        "video_count": _count_suffixes(videos_root, (".avi",)) if videos_root.exists() else 0,
        "mat_count": _count_suffixes(processed_root, (".mat",)) if processed_root.exists() else 0,
    }


def summarize_babel(babel_root: Path) -> dict[str, int]:
    release_root = babel_root / "babel_v1.0_release"
    payload = {}
    for split_path in sorted(release_root.glob("*.json")):
        payload[split_path.name] = split_path.stat().st_size
    return payload


def summarize_mint(mint_root: Path) -> dict[str, object]:
    csv_path = mint_root / "mint_metadata.csv"
    group_dirs = sorted(path.name for path in mint_root.iterdir() if path.is_dir())
    sequence_dirs = {path.parent for path in mint_root.rglob("muscle_activations.pkl")}
    return {
        "has_metadata_csv": csv_path.exists(),
        "group_dirs": group_dirs,
        "sequence_dirs": len(sequence_dirs),
        "pickle_files": _count_suffixes(mint_root, (".pkl",)),
    }


def build_summary(data_root: Path) -> dict[str, object]:
    return {
        "mint": summarize_mint(data_root / "mint_extracted"),
        "amass": summarize_amass(data_root / "amass"),
        "twente": summarize_twente(data_root / "twente_rgb" / "extracted", data_root / "twente_emg" / "extracted"),
        "babel": summarize_babel(data_root / "babel"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a compact summary of extracted raw datasets.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/metadata/dataset_summary.json"))
    args = parser.parse_args()

    summary = build_summary(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
