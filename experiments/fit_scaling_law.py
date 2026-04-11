"""Append scaling-law manifest rows and fit empirical power laws from JSONL.

Subcommands:
  append  — After each training run, record config + metrics (from Lightning CSV).
  fit     — Fit log(L) = c - alpha log(N) via ordinary least squares; optional bootstrap CIs.

Example:
  python experiments/fit_scaling_law.py fit \\
    --manifest experiments/results/scaling_manifest.jsonl \\
    --n-key model/total_params --l-key val_loss_min \\
    --output-json experiments/results/scaling_law_fit.json \\
    --plot experiments/results/scaling_law_fit.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import linregress


def find_latest_metrics_csv(log_root: Path, run_name: str) -> Path | None:
    run_dir = log_root / run_name
    if not run_dir.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for p in run_dir.iterdir():
        if not p.is_dir() or not p.name.startswith("version_"):
            continue
        suffix = p.name.removeprefix("version_")
        try:
            vi = int(suffix)
        except ValueError:
            vi = -1
        csv_path = p / "metrics.csv"
        if csv_path.is_file():
            cand = (vi, csv_path)
            if best is None or cand[0] > best[0]:
                best = cand
    return best[1] if best else None


def parse_lightning_metrics(csv_path: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    if not rows:
        return {
            "val_loss_min": None,
            "val_loss_final": None,
            "val_loss_recent_slope": None,
            "val_loss_still_descending": None,
            "train_loss_final": None,
            "val_train_gap": None,
            "trainer/global_step": None,
            "model/total_params": None,
            "model/trainable_params": None,
        }

    val_key = "val/loss"
    train_epoch_key = "train/loss_epoch"
    total_key = "model/total_params"
    train_key = "model/trainable_params"

    def to_float(x: str | None) -> float | None:
        if x is None or x.strip() == "" or x.lower() == "nan":
            return None
        try:
            return float(x)
        except ValueError:
            return None

    val_losses = [to_float(r.get(val_key)) for r in rows]
    val_losses_n = [v for v in val_losses if v is not None]
    val_loss_min = min(val_losses_n) if val_losses_n else None
    val_loss_final = val_losses_n[-1] if val_losses_n else None

    val_points: list[tuple[float, float]] = []
    for row in rows:
        loss = to_float(row.get(val_key))
        if loss is None:
            continue
        step = (
            to_float(row.get("trainer/global_step"))
            or to_float(row.get("step"))
            or to_float(row.get("global_step"))
        )
        if step is None:
            step = float(len(val_points))
        val_points.append((float(step), float(loss)))

    val_loss_recent_slope = None
    val_loss_still_descending = None
    if len(val_points) >= 4:
        tail_count = max(4, math.ceil(0.2 * len(val_points)))
        tail = val_points[-tail_count:]
        xs = np.array([p[0] for p in tail], dtype=np.float64)
        ys = np.array([p[1] for p in tail], dtype=np.float64)
        if float(xs.max() - xs.min()) > 0:
            reg = linregress(xs, ys)
            val_loss_recent_slope = float(reg.slope)
            val_loss_still_descending = bool(reg.slope < 0.0)

    train_epoch_losses = [to_float(r.get(train_epoch_key)) for r in rows]
    train_epoch_losses_n = [v for v in train_epoch_losses if v is not None]
    train_loss_final = train_epoch_losses_n[-1] if train_epoch_losses_n else None
    val_train_gap = None
    if val_loss_final is not None and train_loss_final is not None:
        val_train_gap = float(train_loss_final) - float(val_loss_final)

    global_steps = [
        to_float(r.get("trainer/global_step")) or to_float(r.get("step")) or to_float(r.get("global_step"))
        for r in rows
    ]
    global_steps_n = [v for v in global_steps if v is not None]
    global_step_final = int(global_steps_n[-1]) if global_steps_n else None

    total_v = train_v = None
    for r in reversed(rows):
        if total_v is None:
            total_v = to_float(r.get(total_key))
        if train_v is None:
            train_v = to_float(r.get(train_key))
        if total_v is not None and train_v is not None:
            break

    return {
        "val_loss_min": val_loss_min,
        "val_loss_final": val_loss_final,
        "val_loss_recent_slope": val_loss_recent_slope,
        "val_loss_still_descending": val_loss_still_descending,
        "train_loss_final": train_loss_final,
        "val_train_gap": val_train_gap,
        "trainer/global_step": global_step_final,
        "model/total_params": total_v,
        "model/trainable_params": train_v,
    }


def metrics_get(row: dict[str, Any], key: str) -> Any:
    metrics = row.get("metrics", {})
    if key in metrics:
        return metrics.get(key)
    if "/" not in key:
        return metrics.get(key)
    head, tail = key.split("/", 1)
    if head == "compute":
        return row.get("compute", {}).get(tail) or metrics.get(key)
    return metrics.get(key)


def build_compute_metrics_block(args: argparse.Namespace, trainable_params: float | None) -> dict[str, Any]:
    ts = int(args.training_max_steps) if args.training_max_steps is not None else None
    flops_per_step = (
        float(args.train_flops_per_optimizer_step) if args.train_flops_per_optimizer_step is not None else None
    )
    trainable_per_sample = (
        float(args.trainable_forward_flops_per_sample) if args.trainable_forward_flops_per_sample is not None else None
    )
    eff_bs = int(args.effective_batch_size) if args.effective_batch_size is not None else None
    ref_flops_per_step = (
        float(args.ref_train_flops_per_optimizer_step) if args.ref_train_flops_per_optimizer_step is not None else None
    )

    total_train_flops = float(ts) * flops_per_step if ts is not None and flops_per_step is not None else None
    trainable_train_flops = None
    if ts is not None and trainable_per_sample is not None and eff_bs is not None:
        trainable_train_flops = float(ts) * (3.0 * trainable_per_sample * float(eff_bs))

    flop_adjusted_trainable_params = None
    flops_per_trainable_param = None
    if trainable_params is not None and flops_per_step is not None and ref_flops_per_step is not None and ref_flops_per_step > 0:
        intensity = flops_per_step / ref_flops_per_step
        flop_adjusted_trainable_params = float(trainable_params) * intensity
    if total_train_flops is not None and trainable_params is not None and trainable_params > 0:
        flops_per_trainable_param = total_train_flops / float(trainable_params)

    return {
        "total_train_flops": total_train_flops,
        "trainable_train_flops": trainable_train_flops,
        "flop_adjusted_trainable_params": flop_adjusted_trainable_params,
        "flops_per_trainable_param": flops_per_trainable_param,
    }


def _build_fairness_block(args: argparse.Namespace) -> dict[str, Any]:
    """Document the training-budget policy used for each scaling-law run."""
    if args.training_max_epochs is None and args.training_max_steps is None and args.compute_mode is None:
        return {
            "compute_budget_match_enabled": False,
            "note": "No per-run training schedule passed to append.",
        }
    te = int(args.training_max_epochs) if args.training_max_epochs is not None else None
    ts = int(args.training_max_steps) if args.training_max_steps is not None else None
    mode = str(args.compute_mode) if args.compute_mode is not None else "fixed"
    matched = mode != "fixed"
    eb = int(args.epoch_baseline) if args.epoch_baseline is not None else None
    eff_bs = int(args.effective_batch_size) if args.effective_batch_size is not None else int(args.batch_size) * int(args.accumulate_grad_batches)
    steps_per_epoch = int(args.optimizer_steps_per_epoch) if args.optimizer_steps_per_epoch is not None else None
    tokens_per_step = (
        int(args.video_tokens_per_optimizer_step) if args.video_tokens_per_optimizer_step is not None else None
    )
    flops_per_step = (
        float(args.train_flops_per_optimizer_step) if args.train_flops_per_optimizer_step is not None else None
    )
    total_tokens = (int(ts) * int(tokens_per_step)) if ts is not None and tokens_per_step is not None else None
    total_flops = (float(ts) * float(flops_per_step)) if ts is not None and flops_per_step is not None else None
    ref_total_tokens = int(args.reference_total_video_tokens) if args.reference_total_video_tokens is not None else None
    ref_total_flops = float(args.reference_total_train_flops) if args.reference_total_train_flops is not None else None
    unclamped_max_steps = float(args.unclamped_max_steps) if args.unclamped_max_steps is not None else None
    rounded_max_steps = int(args.rounded_max_steps) if args.rounded_max_steps is not None else None
    step_min = int(args.step_min) if args.step_min is not None else None
    step_max = int(args.step_max) if args.step_max is not None else None
    budget_was_clamped = bool(int(args.budget_was_clamped)) if args.budget_was_clamped is not None else False
    realized_token_ratio = (float(total_tokens) / float(ref_total_tokens)) if total_tokens is not None and ref_total_tokens not in (None, 0) else None
    realized_flop_ratio = (float(total_flops) / float(ref_total_flops)) if total_flops is not None and ref_total_flops not in (None, 0.0) else None
    return {
        "compute_budget_match_enabled": matched,
        "compute_mode": mode,
        "training_max_epochs": te,
        "training_max_steps": ts,
        "epoch_baseline_from_config": eb,
        "train_samples": int(args.train_samples) if args.train_samples is not None else None,
        "device_count": int(args.device_count) if args.device_count is not None else None,
        "micro_batches_per_epoch": int(args.micro_batches_per_epoch) if args.micro_batches_per_epoch is not None else None,
        "optimizer_steps_per_epoch": steps_per_epoch,
        "effective_batch_size": eff_bs,
        "video_tokens_per_sample": int(args.video_tokens_per_sample) if args.video_tokens_per_sample is not None else None,
        "video_tokens_per_optimizer_step": tokens_per_step,
        "total_forward_flops_per_sample": float(args.total_forward_flops_per_sample) if args.total_forward_flops_per_sample is not None else None,
        "frozen_forward_flops_per_sample": float(args.frozen_forward_flops_per_sample) if args.frozen_forward_flops_per_sample is not None else None,
        "trainable_forward_flops_per_sample": float(args.trainable_forward_flops_per_sample) if args.trainable_forward_flops_per_sample is not None else None,
        "train_flops_per_optimizer_step": flops_per_step,
        "approx_total_video_tokens": total_tokens,
        "approx_total_train_flops": total_flops,
        "realized_vs_reference_video_token_ratio": realized_token_ratio,
        "realized_vs_reference_train_flop_ratio": realized_flop_ratio,
        "reference_video_model_name": args.ref_video_model_name,
        "reference_muscle_hidden_dim": int(args.ref_muscle_hidden_dim) if args.ref_muscle_hidden_dim is not None else None,
        "reference_effective_batch_size": int(args.ref_effective_batch_size) if args.ref_effective_batch_size is not None else None,
        "reference_optimizer_steps_per_epoch": int(args.ref_optimizer_steps_per_epoch) if args.ref_optimizer_steps_per_epoch is not None else None,
        "reference_video_tokens_per_optimizer_step": int(args.ref_video_tokens_per_optimizer_step) if args.ref_video_tokens_per_optimizer_step is not None else None,
        "reference_train_flops_per_optimizer_step": float(args.ref_train_flops_per_optimizer_step) if args.ref_train_flops_per_optimizer_step is not None else None,
        "reference_total_video_tokens": ref_total_tokens,
        "reference_total_train_flops": ref_total_flops,
        "unclamped_max_steps": unclamped_max_steps,
        "rounded_max_steps": rounded_max_steps,
        "step_min": step_min,
        "step_max": step_max,
        "budget_was_clamped": budget_was_clamped,
        "budget_clamp_reason": args.budget_clamp_reason,
        "budget_definition": {
            "video_tokens": "training_max_steps × video_tokens_per_optimizer_step",
            "train_flops": "training_max_steps × train_flops_per_optimizer_step",
        },
        "matching_equation": {
            "fixed": "training_max_steps = epoch_baseline_from_config × optimizer_steps_per_epoch",
            "token_parity": "training_max_steps ≈ reference_total_video_tokens / video_tokens_per_optimizer_step",
            "flop_parity": "training_max_steps ≈ reference_total_train_flops / train_flops_per_optimizer_step",
            "trainable_flop_parity": "training_max_steps ≈ reference_trainable_flops / target_trainable_flops_per_step",
        }.get(mode, "custom"),
        "policy_when_matched": {
            "fixed": "training.max_steps set from the baseline epoch schedule and this run's optimizer_steps_per_epoch.",
            "token_parity": "training.max_steps scaled so total video tokens match the reference run.",
            "flop_parity": "training.max_steps scaled so estimated total training FLOPs match the reference run.",
            "trainable_flop_parity": "training.max_steps scaled so estimated trainable FLOPs (excluding frozen backbone forward) match the reference run.",
        }.get(mode, "custom"),
        "estimation_method": {
            "video_tokens_per_sample": "Derived from VideoMAE patch_size, tubelet_size, configured image_size, and n_frames.",
            "forward_flops": "Measured with torch.utils.flop_counter on the instantiated EgoMuscleModel with dummy inputs.",
            "training_flops": "Frozen forward FLOPs plus 3x trainable forward FLOPs, scaled by effective_batch_size.",
            "schedule": "Rounded optimizer-step target with configured step_min/step_max clamps and max_epochs_cap = ceil(max_steps / optimizer_steps_per_epoch).",
        },
    }


def cmd_append(args: argparse.Namespace) -> None:
    csv_path = find_latest_metrics_csv(Path(args.log_root), args.run_name)
    if csv_path is None:
        print(f"WARN: no metrics.csv for run {args.run_name} under {args.log_root}", flush=True)
        metrics: dict[str, Any] = {
            "val_loss_min": None,
            "model/total_params": None,
            "model/trainable_params": None,
        }
    else:
        metrics = parse_lightning_metrics(csv_path)

    twente_rho = None
    if args.twente_json:
        p = Path(args.twente_json)
        if p.is_file():
            twente_rho = json.loads(p.read_text()).get("mean_rho")

    trainable_params = metrics.get("model/trainable_params")
    compute_block = build_compute_metrics_block(args, trainable_params)

    row = {
        "schema": "scaling_law_v1",
        "run_name": args.run_name,
        "protocol_name": args.protocol_name,
        "design_key": args.design_key,
        "seed": int(args.seed) if args.seed is not None else None,
        "backbone_tag": args.backbone_tag,
        "ladder_mult": float(args.ladder_mult) if getattr(args, "ladder_mult", None) is not None else None,
        "N_definitions": {
            "total_params": (
                "EgoMuscleModel total weights including frozen VideoMAE (training log: model/total_params)."
            ),
            "trainable_params": "Subset with requires_grad=True (training log: model/trainable_params).",
        },
        "L_definitions": {
            "val_loss_min": "Minimum val/loss over logged epochs (AMASS val split).",
            "L_twente": "If twente_mean_rho is present: 1 - mean_rho (blueprint-style normalized error).",
        },
        "model": {
            "video_model_name": args.video_model_name,
            "muscle_hidden_dim": int(args.muscle_hidden_dim),
            "video_trainable_strategy": args.video_trainable_strategy,
            "video_trainable_layers": int(args.video_trainable_layers) if args.video_trainable_layers is not None else None,
            "video_unfreeze_embeddings": bool(int(args.video_unfreeze_embeddings)) if args.video_unfreeze_embeddings is not None else None,
            "fusion_mode": args.fusion_mode,
            "use_video": True,
            "use_muscle": True,
        },
        "compute": {
            "batch_size": int(args.batch_size),
            "accumulate_grad_batches": int(args.accumulate_grad_batches),
            "effective_batch_size": int(args.effective_batch_size) if args.effective_batch_size is not None else int(args.batch_size) * int(args.accumulate_grad_batches),
            "virtual_sampling": bool(int(args.virtual_sampling)) if args.virtual_sampling is not None else False,
            "consumed_train_examples": int(args.consumed_train_examples) if args.consumed_train_examples is not None else None,
            "exposure_multiplier": float(args.exposure_multiplier) if args.exposure_multiplier is not None else None,
            **compute_block,
        },
        "fairness": _build_fairness_block(args),
        "metrics": {
            **metrics,
            "compute/total_train_flops": compute_block.get("total_train_flops"),
            "compute/trainable_train_flops": compute_block.get("trainable_train_flops"),
            "compute/flop_adjusted_trainable_params": compute_block.get("flop_adjusted_trainable_params"),
            "compute/flops_per_trainable_param": compute_block.get("flops_per_trainable_param"),
            "twente_mean_rho": twente_rho,
            "L_twente": (1.0 - float(twente_rho)) if twente_rho is not None else None,
        },
        "paths": {"metrics_csv": str(csv_path) if csv_path else None},
        "logging": {
            "use_wandb": bool(int(args.use_wandb)) if args.use_wandb is not None else False,
            "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity,
            "wandb_group": args.wandb_group,
            "wandb_mode": args.wandb_mode,
        },
    }

    out_path = Path(args.manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    print(f"Appended manifest row for {args.run_name} -> {out_path}", flush=True)


def fit_power_law(
    n_vals: list[float],
    l_vals: list[float],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    pairs = [(n, l) for n, l in zip(n_vals, l_vals) if n > 0 and l > 0 and math.isfinite(n) and math.isfinite(l)]
    if len(pairs) < 3:
        return {"error": "need at least 3 finite (N, L) points with N>0 and L>0"}

    rng = random.Random(seed)
    lx = np.log(np.array([p[0] for p in pairs], dtype=np.float64))
    ly = np.log(np.array([p[1] for p in pairs], dtype=np.float64))
    if len(np.unique(lx)) < 2:
        return {"error": "Cannot calculate a linear regression if all x values are identical"}

    reg = linregress(lx, ly)
    alpha = -float(reg.slope)

    boot_alphas: list[float] = []
    n_arr = np.array([p[0] for p in pairs], dtype=np.float64)
    log_l_arr = np.log(np.array([p[1] for p in pairs], dtype=np.float64))
    for _ in range(max(n_bootstrap, 1)):
        idx = [rng.randrange(len(pairs)) for _ in range(len(pairs))]
        lx_b = np.log(n_arr[idx])
        ly_b = log_l_arr[idx]
        if len(np.unique(lx_b)) < 2:
            continue
        try:
            rb = linregress(lx_b, ly_b)
            boot_alphas.append(-float(rb.slope))
        except ValueError:
            continue

    boot_alphas.sort()
    if len(boot_alphas) >= 2:
        lo = boot_alphas[int(0.025 * len(boot_alphas))]
        hi = boot_alphas[min(int(0.975 * len(boot_alphas)), len(boot_alphas) - 1)]
    else:
        lo = hi = alpha

    return {
        "n_points": len(pairs),
        "alpha": alpha,
        "alpha_ci95_approx": [lo, hi],
        "intercept_log": float(reg.intercept),
        "r_value": float(reg.rvalue),
        "p_value": float(reg.pvalue),
        "stderr_slope": float(reg.stderr) if reg.stderr is not None and not math.isnan(reg.stderr) else None,
    }


def row_passes_filters(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.protocol_name and row.get("protocol_name") != args.protocol_name:
        return False
    if args.filter_hidden is not None:
        hidden = row.get("model", {}).get("muscle_hidden_dim")
        if hidden is None or int(hidden) != int(args.filter_hidden):
            return False
    if args.exclude_clamped and row.get("fairness", {}).get("budget_was_clamped"):
        return False
    if args.exclude_ladder and row.get("ladder_mult") is not None:
        return False
    if args.exclude_under_converged:
        fairness = row.get("fairness", {})
        metrics = row.get("metrics", {})
        max_steps = fairness.get("training_max_steps")
        global_step = metrics.get("trainer/global_step")
        if max_steps is not None and global_step is not None:
            if float(global_step) < 0.95 * float(max_steps):
                return False
        if metrics.get("val_loss_still_descending") is True:
            return False
    return True


def cmd_fit(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    rows: list[dict[str, Any]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    deduped: dict[tuple[Any, Any, Any, Any], dict[str, Any]] = {}
    for row in rows:
        key = (
            row.get("protocol_name"),
            row.get("design_key"),
            row.get("seed"),
            row.get("run_name"),
        )
        deduped[key] = row
    rows = [r for r in deduped.values() if row_passes_filters(r, args)]

    n_key = args.n_key
    l_key = args.l_key

    grouped: dict[str, list[tuple[float, float, str]]] = {}
    for r in rows:
        n = metrics_get(r, n_key)
        l = metrics_get(r, l_key)
        if n is None or l is None:
            continue
        label_value = r.get(args.aggregate_by, "") if args.aggregate_by else ""
        label = str(label_value or r.get("run_name", ""))
        grouped.setdefault(label, []).append((float(n), float(l), str(r.get("run_name", ""))))

    n_vals: list[float] = []
    l_vals: list[float] = []
    labels: list[str] = []
    points: list[dict[str, Any]] = []
    for label, entries in grouped.items():
        if len(entries) < args.min_group_size:
            continue
        ns = [item[0] for item in entries]
        ls = [item[1] for item in entries]
        n_value = statistics.mean(ns)
        if args.l_stat == "mean":
            l_value = statistics.mean(ls)
        elif args.l_stat == "median":
            l_value = statistics.median(ls)
        else:
            l_value = min(ls)
        n_vals.append(float(n_value))
        l_vals.append(float(l_value))
        labels.append(label)
        points.append(
            {
                "label": label,
                "N": float(n_value),
                "L": float(l_value),
                "num_runs": len(entries),
                "L_mean": float(statistics.mean(ls)),
                "L_std": float(statistics.stdev(ls)) if len(ls) >= 2 else 0.0,
                "runs": [item[2] for item in entries],
            }
        )

    fit = fit_power_law(n_vals, l_vals, n_bootstrap=args.bootstrap, seed=args.seed)
    out: dict[str, Any] = {
        "fit": fit,
        "n_key": n_key,
        "l_key": l_key,
        "aggregate_by": args.aggregate_by,
        "l_stat": args.l_stat,
        "points": points,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))

    if args.plot and n_vals:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(n_vals, l_vals, marker="o")
        for i, lab in enumerate(labels):
            ax.annotate(lab, (n_vals[i], l_vals[i]), fontsize=6, xytext=(4, 4), textcoords="offset points")
        if "error" not in fit:
            n_grid = np.linspace(min(n_vals), max(n_vals), 50)
            c = fit["intercept_log"]
            a = fit["alpha"]
            ax.plot(n_grid, np.exp(c) * np.power(n_grid, -a), color="C1", label=rf"$\alpha$={a:.3f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(n_key)
        ax.set_ylabel(l_key)
        ax.legend()
        ax.grid(True, which="both", ls=":")
        plt.tight_layout()
        plt.savefig(args.plot, dpi=150)
        plt.close()
        print(f"Wrote plot {args.plot}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scaling law manifest + log-log fit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("append", help="Append one JSONL manifest row for a completed run")
    pa.add_argument("--run-name", required=True)
    pa.add_argument("--manifest", required=True)
    pa.add_argument("--protocol-name", default="scaling_backbone_trainable_v1")
    pa.add_argument("--design-key", required=True)
    pa.add_argument("--seed", type=int, default=None)
    pa.add_argument("--backbone-tag", default=None)
    pa.add_argument("--log-root", type=Path, default=Path("lightning_logs"))
    pa.add_argument("--video-model-name", required=True)
    pa.add_argument("--muscle-hidden-dim", type=int, required=True)
    pa.add_argument("--video-trainable-strategy", default="frozen")
    pa.add_argument("--video-trainable-layers", type=int, default=None)
    pa.add_argument("--video-unfreeze-embeddings", choices=("0", "1"), default=None)
    pa.add_argument("--batch-size", type=int, required=True)
    pa.add_argument("--accumulate-grad-batches", type=int, required=True)
    pa.add_argument("--fusion-mode", default="cross_attn")
    pa.add_argument("--twente-json", type=Path, default=None)
    pa.add_argument(
        "--training-max-epochs",
        type=int,
        default=None,
        help="Actual trainer max_epochs cap used for this run.",
    )
    pa.add_argument(
        "--training-max-steps",
        type=int,
        default=None,
        help="Actual trainer max_steps used for this run.",
    )
    pa.add_argument(
        "--compute-mode",
        choices=("fixed", "token_parity", "flop_parity", "trainable_flop_parity"),
        default=None,
        help="Compute parity policy used for this run.",
    )
    pa.add_argument(
        "--train-samples",
        type=int,
        default=None,
    )
    pa.add_argument(
        "--device-count",
        type=int,
        default=None,
    )
    pa.add_argument(
        "--micro-batches-per-epoch",
        type=int,
        default=None,
    )
    pa.add_argument(
        "--optimizer-steps-per-epoch",
        type=int,
        default=None,
    )
    pa.add_argument(
        "--video-tokens-per-sample",
        type=int,
        default=None,
    )
    pa.add_argument(
        "--effective-batch-size",
        type=int,
        default=None,
    )
    pa.add_argument(
        "--video-tokens-per-optimizer-step",
        type=int,
        default=None,
    )
    pa.add_argument(
        "--total-forward-flops-per-sample",
        type=float,
        default=None,
    )
    pa.add_argument(
        "--frozen-forward-flops-per-sample",
        type=float,
        default=None,
    )
    pa.add_argument(
        "--trainable-forward-flops-per-sample",
        type=float,
        default=None,
    )
    pa.add_argument(
        "--train-flops-per-optimizer-step",
        type=float,
        default=None,
    )
    pa.add_argument(
        "--ref-video-model-name",
        default=None,
    )
    pa.add_argument(
        "--ref-muscle-hidden-dim",
        type=int,
        default=None,
    )
    pa.add_argument(
        "--ref-effective-batch-size",
        type=int,
        default=None,
    )
    pa.add_argument(
        "--ref-optimizer-steps-per-epoch",
        type=int,
        default=None,
    )
    pa.add_argument(
        "--ref-video-tokens-per-optimizer-step",
        type=int,
        default=None,
    )
    pa.add_argument(
        "--ref-train-flops-per-optimizer-step",
        type=float,
        default=None,
    )
    pa.add_argument("--reference-total-video-tokens", type=int, default=None)
    pa.add_argument("--reference-total-train-flops", type=float, default=None)
    pa.add_argument("--unclamped-max-steps", type=float, default=None)
    pa.add_argument("--rounded-max-steps", type=int, default=None)
    pa.add_argument("--step-min", type=int, default=None)
    pa.add_argument("--step-max", type=int, default=None)
    pa.add_argument("--budget-was-clamped", choices=("0", "1"), default=None)
    pa.add_argument("--budget-clamp-reason", default=None)
    pa.add_argument(
        "--epoch-baseline",
        type=int,
        default=None,
        help="training.max_epochs from config before compute matching (reference schedule).",
    )
    pa.add_argument("--virtual-sampling", choices=("0", "1"), default=None)
    pa.add_argument("--consumed-train-examples", type=int, default=None)
    pa.add_argument("--exposure-multiplier", type=float, default=None)
    pa.add_argument("--use-wandb", choices=("0", "1"), default=None)
    pa.add_argument("--wandb-project", default=None)
    pa.add_argument("--wandb-entity", default=None)
    pa.add_argument("--wandb-group", default=None)
    pa.add_argument("--wandb-mode", default=None)
    pa.set_defaults(func=cmd_append)

    pf = sub.add_parser("fit", help="Fit log L vs log N from manifest JSONL")
    pf.add_argument("--manifest", type=Path, required=True)
    pf.add_argument("--n-key", default="model/trainable_params", help="Key under each row's metrics object")
    pf.add_argument("--l-key", default="val_loss_min", help="Key under each row's metrics object")
    pf.add_argument("--aggregate-by", default="design_key")
    pf.add_argument("--l-stat", choices=("mean", "median", "min"), default="mean")
    pf.add_argument("--min-group-size", type=int, default=1)
    pf.add_argument("--output-json", type=Path, default=Path("experiments/results/scaling_law_fit.json"))
    pf.add_argument("--bootstrap", type=int, default=500)
    pf.add_argument("--seed", type=int, default=0)
    pf.add_argument("--plot", type=Path, default=None)
    pf.add_argument("--protocol-name", default=None, help="Only include rows with this protocol_name.")
    pf.add_argument("--filter-hidden", type=int, default=None, help="Only include rows with model.muscle_hidden_dim equal to this value.")
    pf.add_argument("--exclude-clamped", action="store_true", help="Drop rows where fairness.budget_was_clamped is true.")
    pf.add_argument("--exclude-ladder", action="store_true", help="Drop rows with ladder_mult set (compute-ladder supplement runs).")
    pf.add_argument("--exclude-under-converged", action="store_true", help="Drop rows that did not reach 95%% of max_steps or val_loss_still_descending.")
    pf.set_defaults(func=cmd_fit)

    pm = sub.add_parser("multi-fit", help="Run A1/A2/A3 fits for token-parity backbone sweep (h=256)")
    pm.add_argument("--manifest", type=Path, required=True)
    pm.add_argument("--l-key", default="val_loss_min")
    pm.add_argument("--filter-hidden", type=int, default=256)
    pm.add_argument("--protocol-name", default="paper_scaling_token_parity_flop_reported_v1")
    pm.add_argument("--exclude-clamped", action="store_true", default=True)
    pm.add_argument("--exclude-ladder", action="store_true", default=True)
    pm.add_argument("--exclude-under-converged", action="store_true", default=True)
    pm.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    pm.add_argument("--bootstrap", type=int, default=500)
    pm.add_argument("--seed", type=int, default=0)
    pm.set_defaults(func=cmd_multi_fit)

    pa.add_argument("--ladder-mult", type=float, default=None, help="Compute-ladder step multiplier when applicable.")

    return parser


def cmd_multi_fit(args: argparse.Namespace) -> None:
    fits = [
        ("scaling_law_fit_A1_trainable_params", "model/trainable_params"),
        ("scaling_law_fit_A2_total_train_flops", "compute/total_train_flops"),
        ("scaling_law_fit_A3_flop_adjusted_params", "compute/flop_adjusted_trainable_params"),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stem, n_key in fits:
        fit_args = argparse.Namespace(
            manifest=args.manifest,
            n_key=n_key,
            l_key=args.l_key,
            aggregate_by="design_key",
            l_stat="mean",
            min_group_size=3,
            output_json=args.output_dir / f"{stem}.json",
            bootstrap=args.bootstrap,
            seed=args.seed,
            plot=args.output_dir / f"{stem}.png",
            protocol_name=args.protocol_name,
            filter_hidden=args.filter_hidden,
            exclude_clamped=args.exclude_clamped,
            exclude_ladder=args.exclude_ladder,
            exclude_under_converged=args.exclude_under_converged,
        )
        cmd_fit(fit_args)
    print(f"Wrote multi-fit outputs under {args.output_dir}", flush=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
