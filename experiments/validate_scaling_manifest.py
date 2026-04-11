"""Validate scaling-law manifest rows for paper protocol compliance."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def check_row(row: dict[str, Any], *, warmup_fraction: float, val_stable_eps: float) -> list[str]:
    errors: list[str] = []
    fairness = row.get("fairness", {})
    metrics = row.get("metrics", {})

    if fairness.get("budget_was_clamped"):
        errors.append(f"{row.get('run_name')}: budget_was_clamped=true")

    flop_ratio = fairness.get("realized_vs_reference_train_flop_ratio")
    if flop_ratio is None:
        errors.append(f"{row.get('run_name')}: missing realized_vs_reference_train_flop_ratio")
    elif not math.isfinite(float(flop_ratio)):
        errors.append(f"{row.get('run_name')}: non-finite realized_vs_reference_train_flop_ratio")

    token_ratio = fairness.get("realized_vs_reference_video_token_ratio")
    if token_ratio is not None:
        if abs(float(token_ratio) - 1.0) > 0.05 and row.get("ladder_mult") is None:
            errors.append(
                f"{row.get('run_name')}: video token ratio {token_ratio} not near 1.0 (non-ladder run)"
            )

    max_steps = fairness.get("training_max_steps")
    global_step = metrics.get("trainer/global_step")
    if max_steps is not None and global_step is not None:
        if float(global_step) < 0.95 * float(max_steps):
            errors.append(
                f"{row.get('run_name')}: global_step={global_step} < 95% of training_max_steps={max_steps}"
            )

    warmup_steps = None
    if max_steps is not None:
        warmup_steps = max(1, int(round(warmup_fraction * float(max_steps))))
    if warmup_steps is not None and global_step is not None and float(global_step) <= warmup_steps:
        errors.append(f"{row.get('run_name')}: global_step={global_step} still in warmup (<= {warmup_steps})")

    val_min = metrics.get("val_loss_min")
    val_final = metrics.get("val_loss_final")
    if val_min is not None and val_final is not None:
        if float(val_final) - float(val_min) > val_stable_eps:
            errors.append(
                f"{row.get('run_name')}: val not stable (final-min={(float(val_final) - float(val_min)):.4f} > {val_stable_eps})"
            )

    for key in (
        "compute/total_train_flops",
        "compute/flop_adjusted_trainable_params",
    ):
        if metrics.get(key) is None and row.get("compute", {}).get(key.split("/", 1)[1]) is None:
            errors.append(f"{row.get('run_name')}: missing {key}")

    return errors


def check_schedule_uniformity(rows: list[dict[str, Any]], *, require_same_steps: bool) -> list[str]:
    errors: list[str] = []
    primary = [r for r in rows if r.get("ladder_mult") is None]
    if not primary:
        return errors
    steps = {r.get("fairness", {}).get("training_max_steps") for r in primary}
    steps.discard(None)
    if require_same_steps and len(steps) > 1:
        errors.append(f"primary runs disagree on training_max_steps: {sorted(steps)}")

    eff_bs = {
        r.get("compute", {}).get("effective_batch_size") or r.get("fairness", {}).get("effective_batch_size")
        for r in primary
    }
    eff_bs.discard(None)
    if require_same_steps and len(eff_bs) > 1:
        errors.append(f"primary runs disagree on effective_batch_size: {sorted(eff_bs)}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate scaling manifest JSONL.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--require-same-steps", action="store_true", help="All non-ladder rows must share training_max_steps.")
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--val-stable-eps", type=float, default=0.005)
    parser.add_argument("--min-rows", type=int, default=0)
    args = parser.parse_args()

    rows = load_rows(args.manifest)
    if args.min_rows and len(rows) < args.min_rows:
        print(f"ERROR: expected at least {args.min_rows} rows, found {len(rows)}", file=sys.stderr)
        sys.exit(1)

    all_errors: list[str] = []
    all_errors.extend(check_schedule_uniformity(rows, require_same_steps=args.require_same_steps))
    for row in rows:
        all_errors.extend(check_row(row, warmup_fraction=args.warmup_fraction, val_stable_eps=args.val_stable_eps))

    if all_errors:
        print(f"Manifest validation FAILED ({len(all_errors)} issue(s)):", file=sys.stderr)
        for err in all_errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)

    print(f"Manifest validation OK: {len(rows)} row(s) in {args.manifest}")


if __name__ == "__main__":
    main()
