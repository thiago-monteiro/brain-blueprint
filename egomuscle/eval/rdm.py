from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform


def compute_rdm(representations: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(representations, torch.Tensor):
        representations = representations.detach().cpu().numpy()
    return squareform(pdist(representations, metric="cosine"))


def collect_clip_representations(model, dataloader, device: torch.device) -> dict[str, Any]:
    pooled: list[torch.Tensor] = []
    activities: list[str | None] = []
    clip_ids: list[str] = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            frames = batch["frames"].to(device)
            muscle = None if batch["muscle"] is None else batch["muscle"].to(device)
            activity_ids = batch["activity_id"].to(device)
            outputs = model(frames=frames, muscle=muscle, activity_ids=activity_ids, mask_ratio=0.0)
            pooled.append(outputs.pooled.detach().cpu())
            activities.extend(batch["activity"])
            clip_ids.extend(batch["clip_id"])

    features = torch.cat(pooled, dim=0).numpy()
    return {"features": features, "activities": activities, "clip_ids": clip_ids, "rdm": compute_rdm(features)}


def load_neural_rdms(directory: str | Path) -> dict[str, np.ndarray]:
    root = Path(directory)
    return {path.stem: np.load(path) for path in sorted(root.glob("*.npy"))}
