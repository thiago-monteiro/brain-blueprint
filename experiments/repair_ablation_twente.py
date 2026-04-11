from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_ablations_csv import ABLATIONS, append_jsonl, run_one, write_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun missing/failed Twente evals and rebuild ablation summaries without retraining.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("egomuscle/training/config.yaml"))
    parser.add_argument("--ego-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--exo-root", type=Path, default=Path("data/processed_exo"))
    parser.add_argument("--twente-root", type=Path, default=Path("data/processed_real/twente"))
    parser.add_argument("--summary-name", default="summary.csv")
    parser.add_argument("--jsonl-name", default="summary.jsonl")
    parser.add_argument("--only", nargs="+", choices=[ablation.key for ablation in ABLATIONS])
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--twente-device", default="cuda")
    parser.add_argument("--twente-batch-size", type=int, default=8)
    parser.add_argument("--twente-num-workers", type=int, default=0)
    parser.add_argument("--twente-target-mode", choices=["mean", "flatten"], default="mean")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [ablation for ablation in ABLATIONS if args.only is None or ablation.key in set(args.only)]
    rows = []
    summary_csv = args.root / args.summary_name
    summary_jsonl = args.root / args.jsonl_name
    if summary_jsonl.exists():
        summary_jsonl.unlink()

    for ablation in selected:
        row = run_one(
            ablation,
            config=args.config,
            root=args.root,
            extra_overrides=args.override,
            ego_root=args.ego_root,
            exo_root=args.exo_root,
            twente_root=args.twente_root,
            skip_twente_eval=False,
            twente_device=args.twente_device,
            twente_batch_size=args.twente_batch_size,
            twente_num_workers=args.twente_num_workers,
            twente_target_mode=args.twente_target_mode,
            reuse_existing_train=True,
        )
        rows.append(row)
        append_jsonl(summary_jsonl, row)
        write_summary(summary_csv, rows)
        print(f"[repaired] {ablation.key} twente={row.get('twente_mean_rho', row.get('twente_status', ''))}", flush=True)

    print(f"summary_csv={summary_csv}")
    print(f"summary_jsonl={summary_jsonl}")


if __name__ == "__main__":
    main()
