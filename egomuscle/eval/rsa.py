from __future__ import annotations

import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


def rsa_score(model_rdm: np.ndarray, neural_rdm: np.ndarray) -> tuple[float, float]:
    model_vec = squareform(model_rdm, checks=False)
    neural_vec = squareform(neural_rdm, checks=False)
    score, p_value = spearmanr(model_vec, neural_vec)
    return float(score), float(p_value)


def bootstrap_rsa(
    model_rdm: np.ndarray,
    neural_rdm: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    model_vec = squareform(model_rdm, checks=False)
    neural_vec = squareform(neural_rdm, checks=False)
    n = len(model_vec)
    scores = np.empty(n_bootstrap, dtype=np.float32)

    for idx in range(n_bootstrap):
        sample_indices = rng.integers(0, n, size=n)
        scores[idx] = spearmanr(model_vec[sample_indices], neural_vec[sample_indices]).statistic

    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "ci_low": float(np.quantile(scores, 0.025)),
        "ci_high": float(np.quantile(scores, 0.975)),
    }
