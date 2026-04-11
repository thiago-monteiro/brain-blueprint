from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


def activity_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
) -> dict[str, float]:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    val_x = scaler.transform(val_x)
    
    clf = LogisticRegression(max_iter=2000, multi_class="auto")
    clf.fit(train_x, train_y)
    pred = clf.predict(val_x)
    return {"top1_accuracy": float(accuracy_score(val_y, pred))}


def muscle_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
) -> dict[str, float]:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    val_x = scaler.transform(val_x)
    
    reg = MultiOutputRegressor(Ridge(alpha=1.0))
    reg.fit(train_x, train_y)
    pred = reg.predict(val_x)
    per_group = [r2_score(val_y[:, i], pred[:, i]) for i in range(val_y.shape[1])]
    return {"mean_r2": float(np.mean(per_group))}


def temporal_probe(
    train_x: np.ndarray,
    train_positions: np.ndarray,
    val_x: np.ndarray,
    val_positions: np.ndarray,
) -> dict[str, float]:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    val_x = scaler.transform(val_x)
    
    reg = Ridge(alpha=1.0)
    reg.fit(train_x, train_positions)
    pred = reg.predict(val_x)
    rho, _ = spearmanr(pred, val_positions)
    return {"spearman_rho": float(rho)}


def run_probe_suite(
    train_features: np.ndarray,
    val_features: np.ndarray,
    *,
    train_activity: np.ndarray | None = None,
    val_activity: np.ndarray | None = None,
    train_muscle: np.ndarray | None = None,
    val_muscle: np.ndarray | None = None,
    train_positions: np.ndarray | None = None,
    val_positions: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    if train_activity is not None and val_activity is not None:
        results["activity"] = activity_probe(train_features, train_activity, val_features, val_activity)
    if train_muscle is not None and val_muscle is not None:
        results["muscle"] = muscle_probe(train_features, train_muscle, val_features, val_muscle)
    if train_positions is not None and val_positions is not None:
        results["temporal"] = temporal_probe(train_features, train_positions, val_features, val_positions)
    return results
