import os
import sys
import subprocess
from pathlib import Path

def get_env_list(key, default):
    val = os.environ.get(key, default)
    return [v for v in val.split() if v]

def find_checkpoint(seed_root: Path, ablation: str):
    target_dir = seed_root / ablation
    ckpts = sorted(target_dir.rglob("*.ckpt"))
    if not ckpts:
        print(f"No checkpoint found in {target_dir}", file=sys.stderr)
        sys.exit(1)
    return ckpts[-1]

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    os.chdir(root)

    import argparse
    import random

    parser = argparse.ArgumentParser()
    parser.add_argument("--random-seeds", action="store_true", help="Use 30 random seeds instead of SEEDS")
    args = parser.parse_args()

    run_root = os.environ.get("RUN_ROOT", "experiments/results/clean_runs/paper_clean_v1/ablations_epochs100")
    config = os.environ.get("CONFIG", "egomuscle/training/config.yaml")
    out_root = os.environ.get("OUT_ROOT", "experiments/results/blueprint_followups/paper_clean_v1")
    seeds = get_env_list("SEEDS", "0 1 2")
    if args.random_seeds:
        seeds = [str(random.randint(0, 2**31 - 1)) for _ in range(30)]
    layerwise_ablations = get_env_list("LAYERWISE_ABLATIONS", "E3 E4 E5")
    temporal_ablations = get_env_list("TEMPORAL_ABLATIONS", "E2 E3 E4 E5")
    temporal_offsets = get_env_list("TEMPORAL_OFFSETS", "-8 -4 -2 -1 0 1 2 4 8")
    dataset_root = os.environ.get("DATASET_ROOT", "data/processed")
    ego_root = os.environ.get("EGO_ROOT", "data/processed")
    exo_root = os.environ.get("EXO_ROOT", "data/processed_exo")
    twente_root = os.environ.get("TWENTE_ROOT", "data/processed_real/twente")
    neural_rdm_dir = os.environ.get("NEURAL_RDM_DIR", "egomuscle/eval/neural_rdms")
    neural_mat_root = os.environ.get("NEURAL_MAT_ROOT", "data/raw/bold5000/extracted/BOLD5000_GLMsingle_ROI_betas/mat")
    stimuli_dir = os.environ.get("STIMULI_DIR", "")
    stimuli_list = os.environ.get("STIMULI_LIST", "")

    common_overrides = [
        "--override", "model.video_model_name=OpenGVLab/VideoMAEv2-Base",
        "--override", "model.video_trainable_strategy=last_n",
        "--override", "model.video_trainable_layers=2",
        "--override", "model.video_unfreeze_embeddings=false",
        "--override", "model.muscle_hidden_dim=256",
        "--override", "model.label_vocab_size=1024",
        "--override", "model.fusion_dropout=0.1",
        "--override", "model.pred_dropout=0.1",
    ]

    os.makedirs(out_root, exist_ok=True)

    has_rdms = any(Path(neural_rdm_dir).glob("*.npy"))
    if not has_rdms:
        if os.path.isdir(neural_mat_root):
            print("==================================================================")
            print(" Build Neural RDMs")
            print("==================================================================")
            subprocess.run([
                "python", "-m", "egomuscle.eval.prepare_bold5000_rdms",
                "--mat-root", neural_mat_root,
                "--output-dir", neural_rdm_dir
            ], check=True)
        else:
            print("Missing neural RDM prerequisite.", file=sys.stderr)
            print(f"Expected either precomputed .npy files under {neural_rdm_dir} or BOLD5000 ROI betas under {neural_mat_root}.", file=sys.stderr)
            sys.exit(1)

    print("==================================================================")
    print(" Repair Twente + Rebuild Summaries")
    print("==================================================================")
    for seed in seeds:
        seed_root_str = os.path.join(run_root, f"seed_{seed}")
        print(f"[twente] seed_{seed}")
        subprocess.run([
            "python", "experiments/repair_ablation_twente.py",
            "--root", seed_root_str,
            "--config", config,
            "--ego-root", ego_root,
            "--exo-root", exo_root,
            "--twente-root", twente_root
        ] + common_overrides, check=True)

    print("\n==================================================================")
    print(" Layerwise Hierarchy")
    print("==================================================================")
    layerwise_extra_args = []
    if stimuli_dir:
        layerwise_extra_args.extend(["--stimuli-dir", stimuli_dir])
    if stimuli_list:
        layerwise_extra_args.extend(["--stimuli-list", stimuli_list])

    for seed in seeds:
        seed_root_path = Path(run_root) / f"seed_{seed}"
        for ablation in layerwise_ablations:
            ckpt = find_checkpoint(seed_root_path, ablation)
            print(f"[layerwise] seed_{seed} {ablation}")
            out_file = os.path.join(out_root, f"layerwise_seed_{seed}_{ablation}.json")
            subprocess.run([
                "python", "experiments/run_layerwise_hierarchy.py",
                "--checkpoint", str(ckpt),
                "--config", config,
                "--split", "val",
                "--neural-dir", neural_rdm_dir,
                "--output", out_file
            ] + layerwise_extra_args + common_overrides, check=True)

    print("\n==================================================================")
    print(" Temporal Alignment Sweep")
    print("==================================================================")
    for seed in seeds:
        seed_root_str = os.path.join(run_root, f"seed_{seed}")
        print(f"[temporal] seed_{seed}")
        out_file = os.path.join(out_root, f"temporal_alignment_seed_{seed}.json")
        cmd = [
            "python", "experiments/run_temporal_alignment_sweep.py",
            "--ablation-root", seed_root_str,
            "--config", config,
            "--dataset-root", dataset_root,
            "--split", "val",
            "--only"
        ] + temporal_ablations + ["--offsets"] + temporal_offsets + ["--output", out_file] + common_overrides
        subprocess.run(cmd, check=True)

    print("\nDone.")
    print(f"Input run root: {run_root}")
    print(f"Outputs: {out_root}")

if __name__ == "__main__":
    main()
