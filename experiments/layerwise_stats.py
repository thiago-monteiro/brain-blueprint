from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
from scipy.stats import rankdata, spearmanr

logger = logging.getLogger("layerwise_rsa")


def resolve_stats_workers(requested: int) -> int:
    if requested > 0:
        return requested
    cpu = os.cpu_count() or 1
    return max(1, cpu)


def rank_vector(values: np.ndarray) -> np.ndarray:
    return rankdata(np.asarray(values, dtype=np.float64), method="average")


def pearson_on_ranks(rank_a: np.ndarray, rank_b: np.ndarray) -> float:
    a = np.asarray(rank_a, dtype=np.float64)
    b = np.asarray(rank_b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum())) + 1.0e-12
    return float((a * b).sum() / denom)


def fast_spearman_vec(a: np.ndarray, b: np.ndarray) -> float:
    return pearson_on_ranks(rank_vector(a), rank_vector(b))


def fast_spearman_vec_precomputed(a: np.ndarray, rank_b: np.ndarray) -> float:
    return pearson_on_ranks(rank_vector(a), rank_b)


def safe_spearman_vec(a: np.ndarray, b: np.ndarray) -> float:
    rho = spearmanr(a, b).statistic
    return float(0.0 if np.isnan(rho) else rho)


def _permutation_null_value(
    child_seed: int,
    *,
    model_vec: np.ndarray,
    neural_ranks: np.ndarray,
    model_rdm: np.ndarray,
    mode: str,
) -> float:
    rng = np.random.default_rng(child_seed)
    if mode == "mantel":
        n = model_rdm.shape[0]
        perm = rng.permutation(n)
        perm_vec = np.asarray(model_rdm[np.ix_(perm, perm)], dtype=np.float64)
        perm_vec = perm_vec[np.triu_indices(n, k=1)]
        return fast_spearman_vec_precomputed(perm_vec, neural_ranks)
    perm_vec = model_vec[rng.permutation(len(model_vec))]
    return fast_spearman_vec_precomputed(perm_vec, neural_ranks)


def permutation_test_rdm(
    model_rdm: np.ndarray,
    neural_vec: np.ndarray,
    observed: float,
    *,
    n_permutations: int,
    seed: int,
    mode: str,
    n_workers: int = 1,
    neural_ranks: np.ndarray | None = None,
    nested_outer_parallel: bool = False,
    perm_thread_workers: int = 8,
) -> dict[str, float | str]:
    if n_permutations <= 0:
        return {"n_permutations": 0, "mode": mode, "p_greater": float("nan"), "p_two_sided": float("nan")}

    model_vec = np.asarray(model_rdm[np.triu_indices(model_rdm.shape[0], k=1)], dtype=np.float64)
    rank_neural = neural_ranks if neural_ranks is not None else rank_vector(neural_vec)
    workers = resolve_stats_workers(n_workers)
    use_thread_pool = nested_outer_parallel and n_permutations >= 8
    if use_thread_pool:
        workers = max(1, min(int(perm_thread_workers), os.cpu_count() or 1))
        logger.info(
            "permutation_test_rdm: outer joblib active; using threading backend with n_jobs=%s",
            workers,
        )
    elif workers <= 1:
        logger.info("permutation_test_rdm: serial null loop (perm_workers=%s)", workers)

    if n_permutations < 8 or (workers <= 1 and not use_thread_pool):
        rng = np.random.default_rng(seed)
        null = np.empty(n_permutations, dtype=np.float32)
        for idx in range(n_permutations):
            if mode == "mantel":
                n = model_rdm.shape[0]
                perm = rng.permutation(n)
                perm_vec = np.asarray(model_rdm[np.ix_(perm, perm)], dtype=np.float64)
                perm_vec = perm_vec[np.triu_indices(n, k=1)]
            else:
                perm_vec = model_vec[rng.permutation(len(model_vec))]
            null[idx] = fast_spearman_vec_precomputed(perm_vec, rank_neural)
    else:
        from joblib import Parallel, delayed

        child_seeds = [
            int(s.generate_state(1, dtype=np.uint64)[0])
            for s in np.random.SeedSequence(seed).spawn(n_permutations)
        ]
        if use_thread_pool:
            backend = "threading"
        else:
            backend = "threading" if len(model_vec) > 500_000 else "loky"
        null = Parallel(n_jobs=workers, backend=backend)(
            delayed(_permutation_null_value)(
                child_seed,
                model_vec=model_vec,
                neural_ranks=rank_neural,
                model_rdm=model_rdm,
                mode=mode,
            )
            for child_seed in child_seeds
        )
        null = np.asarray(null, dtype=np.float32)

    p_greater = (float(np.sum(null >= observed)) + 1.0) / (float(n_permutations) + 1.0)
    p_two = (float(np.sum(np.abs(null) >= abs(observed))) + 1.0) / (float(n_permutations) + 1.0)
    return {
        "n_permutations": int(n_permutations),
        "mode": mode,
        "p_greater": p_greater,
        "p_two_sided": p_two,
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)) if n_permutations > 1 else 0.0,
    }
