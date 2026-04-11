from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from egomuscle.eval.rdm import compute_rdm
from egomuscle.eval.rsa import rsa_score, bootstrap_rsa
from egomuscle.training.train import EgoMuscleLightningModule, apply_override, load_config


class StaticImageDataset(Dataset):
    """Wraps a list of images as static 16-frame videos for the VideoMAE encoder."""
    def __init__(self, image_paths: list[Path], n_frames: int = 16, image_size: int = 224) -> None:
        self.image_paths = image_paths
        self.n_frames = n_frames
        self.image_size = image_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_tensor = (img_tensor - self.mean) / self.std
        frames = img_tensor.unsqueeze(0).repeat(self.n_frames, 1, 1, 1)
        
        return {
            "frames": frames,
            "clip_id": path.stem,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RSA benchmarks against brain data.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("egomuscle/training/config.yaml"))
    parser.add_argument("--stimuli-dir", type=Path, help="Path to BOLD5000 or Cichy stimuli images.")
    parser.add_argument("--stimuli-list", type=Path, help="Text file with ordered list of image names.")
    parser.add_argument("--neural-dir", type=Path, default=Path("egomuscle/eval/neural_rdms"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/rsa_benchmarks.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading checkpoint from {args.checkpoint}...")
    config = load_config(args.config)
    for override in args.override:
        apply_override(config, override)
    
    module = EgoMuscleLightningModule(config)
    payload = torch.load(args.checkpoint, map_location=device)
    state_dict = payload.get("state_dict", payload)
    module.load_state_dict(state_dict, strict=True)
    module.to(device)
    module.eval()

    if args.stimuli_list and args.stimuli_list.exists():
        lines = args.stimuli_list.read_text().splitlines()
        image_paths = []
        print(f"Resolving {len(lines)} stimuli from list...")
        for line in tqdm(lines):
            line = line.strip()
            if not line: continue
            
            matches = list(args.stimuli_dir.rglob(line))
            if not matches and not line.endswith((".jpg", ".png", ".JPEG")):
                 for ext in [".jpg", ".JPEG", ".png"]:
                     matches = list(args.stimuli_dir.rglob(line + ext))
                     if matches: break
            
            if matches:
                image_paths.append(matches[0])
            else:
                pass
        print(f"Loaded {len(image_paths)} stimuli from list.")
    elif args.stimuli_dir and args.stimuli_dir.exists():
        image_paths = sorted(list(args.stimuli_dir.rglob("*.jpg")) + list(args.stimuli_dir.rglob("*.png")))
        print(f"Found {len(image_paths)} stimuli images in {args.stimuli_dir}")
    else:
        print("Warning: Stimuli directory not found or not provided.")
        image_paths = []

    neural_rdms = {}
    for path in sorted(args.neural_dir.glob("*.npy")):
        neural_rdms[path.stem] = np.load(path)
    
    if not neural_rdms:
        print(f"No neural RDMs found in {args.neural_dir}")
        return

    if image_paths:
        dataset = StaticImageDataset(image_paths)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        reps = []
        print("Extracting model representations for stimuli...")
        with torch.no_grad():
            for batch in tqdm(dataloader):
                frames = batch["frames"].to(device)
                outputs = module.model(frames=frames, muscle=None, activity_ids=None, mask_ratio=0.0)
                reps.append(outputs.pooled.cpu().numpy())
        
        features = np.concatenate(reps, axis=0)
        model_rdm = compute_rdm(features)
    else:
        print("Skipping stimuli extraction.")
        return

    results = {
        "checkpoint": str(args.checkpoint),
        "stimuli_count": int(features.shape[0]),
        "benchmarks": {}
    }

    for region, n_rdm in neural_rdms.items():
        if n_rdm.shape[0] != model_rdm.shape[0]:
            print(f"Skipping {region}: Neural {n_rdm.shape[0]} != Model {model_rdm.shape[0]}")
            if model_rdm.shape[0] < n_rdm.shape[0]:
                print(f"Subsetting {region} to match model size...")
                n_rdm = n_rdm[:model_rdm.shape[0], :model_rdm.shape[0]]
                score, p_val = rsa_score(model_rdm, n_rdm)
                boot = None
            else:
                continue
        else:
            score, p_val = rsa_score(model_rdm, n_rdm)
            boot = None
        
        results["benchmarks"][region] = {
            "spearman_rho": score,
            "p_value": p_val,
            "bootstrap": boot
        }
        print(f"RSA [{region}]: {score:.4f} (p={p_val:.4e})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
