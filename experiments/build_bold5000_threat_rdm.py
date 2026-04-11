from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


THREAT_TERMS = {
    "alligator", "ambulance", "ashcan", "battle", "cemetery", "chemical", "cliff",
    "crash", "crocodile", "fire", "garbage", "gun", "missile", "police", "prison",
    "revolver", "rifle", "scorpion", "shark", "snake", "spider", "tiger", "volcano",
    "weapon", "wreck",
}

NEUTRAL_TERMS = {
    "bakery", "beach", "bedroom", "bookshop", "class", "copyroom", "field", "forest",
    "garden", "kitchen", "library", "livingroom", "mountain", "office", "park",
    "restaurant", "road", "shop", "studio", "store",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a real human threat RDM from BOLD5000 ROI RDMs.")
    parser.add_argument("--stimuli-list", type=Path, default=Path("experiments/results/bold5000_stimuli_list_neural_order.txt"))
    parser.add_argument("--stimuli-root", type=Path, default=Path("data/raw/bold5000/extracted/BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli"))
    parser.add_argument("--label-root", type=Path, default=Path("data/raw/bold5000/extracted/BOLD5000_Stimuli/Image_Labels"))
    parser.add_argument("--neural-rdm", type=Path, default=Path("egomuscle/eval/neural_rdms/inferotemporal.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/bold5000_threat_rdm"))
    parser.add_argument("--max-per-class", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_imagenet_labels(path: Path) -> dict[str, str]:
    labels = {}
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            labels[parts[0]] = parts[1].lower()
    return labels


def text_for_stimulus(rel_path: str, imagenet: dict[str, str]) -> str:
    path = Path(rel_path)
    if rel_path.startswith("ImageNet/"):
        synset = path.name.split("_")[0]
        return imagenet.get(synset, path.stem).lower()
    if rel_path.startswith("Scene/"):
        return re.sub(r"\d+$", "", path.stem).lower()
    return path.stem.lower()


def classify(text: str) -> str | None:
    normalized = re.sub(r"[^a-z]+", " ", text.lower())
    tokens = set(normalized.split())
    joined = normalized.replace(" ", "")
    threat = any(term in tokens or term in joined for term in THREAT_TERMS)
    neutral = any(term in tokens or term in joined for term in NEUTRAL_TERMS)
    if threat and not neutral:
        return "threat"
    if neutral and not threat:
        return "neutral"
    return None


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    imagenet = load_imagenet_labels(args.label_root / "imagenet_final_labels.txt")
    stimuli = [line.strip() for line in args.stimuli_list.read_text().splitlines() if line.strip()]
    if len(stimuli) != 5254:
        raise ValueError(f"Expected 5254 BOLD5000 stimuli, got {len(stimuli)}")

    candidates = []
    seen = set()
    for idx, rel_path in enumerate(stimuli):
        text = text_for_stimulus(rel_path, imagenet)
        cls = classify(text)
        image_path = args.stimuli_root / rel_path
        if cls is None or not image_path.is_file() or rel_path in seen:
            continue
        seen.add(rel_path)
        candidates.append({"row_index": idx, "class": cls, "rel_path": rel_path, "image_path": str(image_path), "label_text": text})

    rng = np.random.default_rng(args.seed)
    selected = []
    for cls in ("threat", "neutral"):
        rows = [row for row in candidates if row["class"] == cls]
        if len(rows) < args.max_per_class:
            raise ValueError(f"Only {len(rows)} {cls} candidates; lower --max-per-class")
        chosen = rng.choice(len(rows), size=args.max_per_class, replace=False)
        selected.extend(rows[int(i)] for i in chosen)
    selected = sorted(selected, key=lambda row: (row["class"], row["rel_path"]))
    indices = np.array([int(row["row_index"]) for row in selected], dtype=np.int64)

    full_rdm = np.load(args.neural_rdm)
    if full_rdm.shape != (5254, 5254):
        raise ValueError(f"Expected 5254x5254 neural RDM, got {full_rdm.shape}")
    threat_rdm = full_rdm[np.ix_(indices, indices)]
    np.save(args.output_dir / "human_threat_rdm.npy", threat_rdm)

    with (args.output_dir / "stimuli_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_index", "class", "rel_path", "image_path", "label_text"])
        writer.writeheader()
        writer.writerows(selected)
    summary = {
        "source": "BOLD5000 GLMsingle ROI RDM",
        "neural_rdm": str(args.neural_rdm),
        "rdm_shape": list(threat_rdm.shape),
        "num_threat": sum(1 for row in selected if row["class"] == "threat"),
        "num_neutral": sum(1 for row in selected if row["class"] == "neutral"),
        "selection_note": "Threat/neutral labels are keyword-derived from BOLD5000 ImageNet and Scene labels; COCO images are excluded unless filename text matches.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
