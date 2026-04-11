from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge SMFE-Memory experiment outputs into paper-facing summary tables.")
    parser.add_argument("--root", type=Path, default=Path("experiments/results"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/blueprint_v2_summary.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("experiments/results/blueprint_v2_summary.csv"))
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def flatten(prefix: str, value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            result.update(flatten(f"{prefix}_{key}" if prefix else str(key), child))
        return result
    if isinstance(value, list):
        return {f"{prefix}_count": len(value)}
    return {prefix: value}


def collect_training_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(root.rglob("metrics.csv")):
        with metrics_path.open(newline="") as handle:
            metrics_rows = list(csv.DictReader(handle))
        if not metrics_rows:
            continue
        best_val = None
        for row in metrics_rows:
            raw = row.get("val/loss", "").strip()
            if not raw:
                continue
            try:
                loss = float(raw)
            except ValueError:
                continue
            if best_val is None or loss < best_val[0]:
                best_val = (loss, row)
        if best_val is None:
            continue
        _, row = best_val
        summary = {
            "source": "training",
            "run": str(metrics_path.parent.parent),
            "metrics_path": str(metrics_path),
        }
        for key, value in row.items():
            if value.strip() and (key.startswith("val/") or key.startswith("train/")):
                summary[key.replace("/", "_")] = value
        rows.append(summary)
    return rows


def collect_json_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    patterns = {
        "memory": "*memory_probe_suite.json",
        "pbit_quant": "*pbit_quantization_sweep.json",
        "ltm": "*ltm_probe_suite.json",
        "temporal": "*temporal_alignment_sweep.json",
        "layerwise": "*layerwise_hierarchy.json",
        "twente": "*twente_eval.json",
        "es": "*selected_config.json",
        "agency": "*agency_boundary_probe.json",
    }
    for source, pattern in patterns.items():
        for path in sorted(root.rglob(pattern)):
            payload = read_json(path)
            if payload is None:
                continue
            row = {"source": source, "path": str(path)}
            if source in {"memory", "ltm", "pbit_quant"} and "rows" in payload:
                row[f"{source}_rows"] = len(payload["rows"])
            row.update(flatten(source, {key: value for key, value in payload.items() if key not in {"rows", "gap_retention", "layers"}}))
            if "gap_retention" in payload:
                for gap_row in payload["gap_retention"]:
                    gap = int(gap_row.get("gap", 0))
                    for key, value in gap_row.items():
                        if key != "gap":
                            row[f"memory_gap_{gap}_{key}"] = value
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = collect_training_rows(args.root) + collect_json_rows(args.root)
    payload = {"root": str(args.root), "num_rows": len(rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    write_csv(args.csv_output, rows)
    print(json.dumps({"num_rows": len(rows), "output": str(args.output), "csv_output": str(args.csv_output)}, indent=2))


if __name__ == "__main__":
    main()
