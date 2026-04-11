from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def aggregate_attention(attention: torch.Tensor, activities: list[str | None]) -> dict[str, np.ndarray]:
    if attention.ndim == 4:
        attention = attention.mean(dim=1)
    activity_attention: dict[str, list[np.ndarray]] = defaultdict(list)
    framewise = attention.mean(dim=1).detach().cpu().numpy()

    for idx, activity in enumerate(activities):
        label = activity or "unknown"
        activity_attention[label].append(framewise[idx])

    return {key: np.stack(values).mean(axis=0) for key, values in activity_attention.items()}


def plot_attention_heatmap(activity_attention: dict[str, np.ndarray], output_path: str | Path) -> None:
    if not activity_attention:
        raise ValueError("No attention values available to plot.")
    activities = sorted(activity_attention)
    matrix = np.stack([activity_attention[activity] for activity in activities], axis=0)

    fig, ax = plt.subplots(figsize=(12, max(4, len(activities) * 0.4)))
    sns.heatmap(matrix, cmap="mako", yticklabels=activities, ax=ax)
    ax.set_xlabel("Conditioning timestep")
    ax.set_ylabel("Activity")
    ax.set_title("Cross-attention anatomy")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
