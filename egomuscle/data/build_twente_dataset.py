from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def extract_emg(mat_path: Path) -> np.ndarray:
    obj = loadmat(mat_path, squeeze_me=True, struct_as_record=False)["Datastr"]
    emg = obj.Resample.EMG
    return np.asarray(emg, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Twente real-EMG evaluation dataset.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifests/twente_pairs.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed_real/twente"))
    parser.add_argument("--view", choices=["front", "side"], default="front")
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    activity_vocab = sorted({str(record["trial"]) for record in records})
    activity_to_id = {activity: idx for idx, activity in enumerate(activity_vocab)}
    (args.output_root / "muscles").mkdir(parents=True, exist_ok=True)
    (args.output_root / "clips").mkdir(parents=True, exist_ok=True)
    metadata = []
    failures = []

    for record in records:
        video_key = f"{args.view}_video"
        video_path = record.get(video_key)
        if video_path is None:
            failures.append({"subject": record["subject"], "trial": record["trial"], "reason": f"missing_{video_key}"})
            continue

        stem = f"{record['subject']}__{record['trial']}"
        muscle = extract_emg(Path(record["emg_mat"]))
        muscle_out = args.output_root / "muscles" / f"{stem}.npy"
        clip_out = args.output_root / "clips" / f"{stem}.avi"
        np.save(muscle_out, muscle)
        symlink_or_copy(Path(video_path), clip_out)
        metadata.append(
            {
                "clip_id": stem,
                "activity": record["trial"],
                "activity_id": activity_to_id[str(record["trial"])],
                "subject": record["subject"],
                "trial": record["trial"],
                "view": args.view,
                "clip_path": str(clip_out),
                "muscle_path": str(muscle_out),
                "emg_dim": int(muscle.shape[1] if muscle.ndim > 1 else 1),
                "num_samples": int(muscle.shape[0]),
            }
        )

    (args.output_root / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (args.output_root / "activity_vocab.json").write_text(json.dumps(activity_to_id, indent=2))
    report = {
        "records": len(records),
        "exported": len(metadata),
        "failures": failures,
        "view": args.view,
        "activity_count": len(activity_vocab),
    }
    (args.output_root / "build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
