from __future__ import annotations

import argparse
import json
from pathlib import Path


def iter_babel_records(babel_root: Path):
    release_root = babel_root / "babel_v1.0_release"
    for split_path in sorted(release_root.glob("*.json")):
        split_name = split_path.stem
        payload = json.loads(split_path.read_text())
        for _, record in payload.items():
            feat_p = record.get("feat_p")
            if feat_p:
                yield split_name, record


def build_babel_index(babel_root: Path) -> dict[str, dict[str, object]]:
    index: dict[str, dict[str, object]] = {}
    for split_name, record in iter_babel_records(babel_root):
        key = record["feat_p"].removesuffix(".npz")
        index[key] = {
            "babel_sid": record.get("babel_sid"),
            "split": split_name,
            "feat_p": record.get("feat_p"),
            "url": record.get("url"),
            "dur": record.get("dur"),
            "seq_ann": record.get("seq_ann"),
            "frame_ann": record.get("frame_ann"),
        }
    return index


def canonical_mint_key(sequence_dir: Path, mint_root: Path) -> str:
    relative = sequence_dir.relative_to(mint_root).as_posix()
    return relative


def build_mint_manifest(mint_root: Path, babel_index: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for activations_path in sorted(mint_root.rglob("muscle_activations.pkl")):
        sequence_dir = activations_path.parent
        key = canonical_mint_key(sequence_dir, mint_root)
        babel = babel_index.get(key)
        records.append(
            {
                "mint_key": key,
                "dataset": sequence_dir.parts[len(mint_root.parts)],
                "subject": sequence_dir.parts[-2] if len(sequence_dir.parts) >= 2 else None,
                "sequence": sequence_dir.name,
                "muscle_activations": str(activations_path),
                "muscle_forces": str(sequence_dir / "muscle_forces.pkl"),
                "grf": str(sequence_dir / "grf.pkl"),
                "babel": babel,
            }
        )
    return records


def index_twente_videos(videos_root: Path) -> dict[tuple[str, str], dict[str, str]]:
    index: dict[tuple[str, str], dict[str, str]] = {}
    if not videos_root.exists():
        return index

    for avi_path in sorted(videos_root.rglob("*.avi")):
        subject = avi_path.parent.name
        stem = avi_path.stem
        if stem.endswith("_front_anonymized"):
            trial = stem.removesuffix("_front_anonymized")
            view = "front"
        elif stem.endswith("_side_anonymized"):
            trial = stem.removesuffix("_side_anonymized")
            view = "side"
        else:
            continue
        index.setdefault((subject, trial), {})[view] = str(avi_path)
    return index


def build_twente_manifest(emg_root: Path, rgb_root: Path) -> list[dict[str, object]]:
    videos_root = rgb_root / "Videos_anonymized"
    processed_root = emg_root / "Processed_data"
    if not processed_root.exists():
        return []

    video_index = index_twente_videos(videos_root)
    records: list[dict[str, object]] = []

    for subject_dir in sorted(path for path in processed_root.iterdir() if path.is_dir() and path.name.startswith("Subj")):
        subject = subject_dir.name
        for mat_path in sorted(subject_dir.glob(f"{subject}_*.mat")):
            trial = mat_path.stem.removeprefix(f"{subject}_")
            video_pair = video_index.get((subject, trial), {})
            records.append(
                {
                    "subject": subject,
                    "trial": trial,
                    "emg_mat": str(mat_path),
                    "front_video": video_pair.get("front"),
                    "side_video": video_pair.get("side"),
                }
            )
    return records


def write_jsonl(records: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build structured manifests for MinT, Twente, and BABEL.")
    parser.add_argument("--mint-root", type=Path, default=Path("data/raw/mint_extracted"))
    parser.add_argument("--babel-root", type=Path, default=Path("data/raw/babel"))
    parser.add_argument("--twente-emg-root", type=Path, default=Path("data/raw/twente_emg/extracted"))
    parser.add_argument("--twente-rgb-root", type=Path, default=Path("data/raw/twente_rgb/extracted"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/manifests"))
    args = parser.parse_args()

    babel_index = build_babel_index(args.babel_root)
    mint_manifest = build_mint_manifest(args.mint_root, babel_index)
    twente_manifest = build_twente_manifest(args.twente_emg_root, args.twente_rgb_root)

    write_jsonl(mint_manifest, args.output_dir / "mint_sequences.jsonl")
    write_jsonl(twente_manifest, args.output_dir / "twente_pairs.jsonl")

    summary = {
        "mint_sequences": len(mint_manifest),
        "mint_with_babel": sum(1 for record in mint_manifest if record["babel"] is not None),
        "twente_pairs": len(twente_manifest),
        "twente_complete_video_pairs": sum(
            1 for record in twente_manifest if record["front_video"] is not None and record["side_video"] is not None
        ),
    }
    (args.output_dir / "manifest_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
