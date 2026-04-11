import math
import os
import sys
import subprocess
import yaml
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

def get_env(key, default=""):
    return os.environ.get(key, default)

def read_training_max_epochs(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return int(cfg["training"]["max_epochs"])

def canonical_backbone_id(tag):
    mapping = {
        "small_ft_kin": "MCG-NJU/videomae-small-finetuned-kinetics",
        "base_ft_kin": "MCG-NJU/videomae-base-finetuned-kinetics",
        "large_ft_kin": "MCG-NJU/videomae-large-finetuned-kinetics",
        "huge_ft_kin": "MCG-NJU/videomae-huge-finetuned-kinetics",
        "small_ssl": "MCG-NJU/videomae-small",
        "base_ssl": "MCG-NJU/videomae-base",
        "large_ssl": "MCG-NJU/videomae-large",
        "huge_ssl": "MCG-NJU/videomae-huge",
        "base_v2": "OpenGVLab/VideoMAEv2-Base",
        "large_v2": "OpenGVLab/VideoMAEv2-Large",
        "huge_v2": "OpenGVLab/VideoMAEv2-Huge",
        "base_hf": "MCG-NJU/videomae-base",
        "base_ft_ssv2": "MCG-NJU/videomae-base-finetuned-ssv2",
        "small_ft_ssv2": "MCG-NJU/videomae-small-finetuned-ssv2",
        "large_hf": "MCG-NJU/videomae-large",
        "large_ft_ssv2": "MCG-NJU/videomae-large-finetuned-ssv2"
    }
    return mapping.get(tag, None)

def resolve_video_model_id(tag, raw):
    trimmed = raw.strip()
    canonical = canonical_backbone_id(tag)
    if canonical:
        if not trimmed or trimmed == tag or "-finetuned-" in trimmed or "/videomae-" in trimmed:
            return canonical
    return trimmed

def validate_video_model_id(tag, vid):
    if not vid or "-finetuned-" in vid or "/videomae-" in vid:
        print(f"ERROR: malformed video model id for tag={tag}: '{vid}'", file=sys.stderr)
        canonical = canonical_backbone_id(tag)
        if canonical:
            print(f"Expected canonical id for {tag}: {canonical}", file=sys.stderr)
        sys.exit(1)

TOKEN_PARITY_EFFECTIVE_BATCH = 128


def vram_batch_accum(tag, hidden):
    """Micro-batch size and grad accumulation. Effective batch = batch * accumulate (× devices).

    VideoMAEv2 Base/Large match v1 param counts but use more activation memory in our hub
    forward (full token sequence, last_n unfrozen blocks). v2 tags therefore use smaller
    micro-batches; accumulate is scaled up so token-parity effective batch stays 128.
    """
    h = int(hidden)
    if tag == "base_v2":
        if h <= 256:
            bs, acc = 16, 8
        else:
            bs, acc = 8, 16
    elif tag == "large_v2":
        if h <= 128:
            bs, acc = 8, 16
        elif h <= 256:
            bs, acc = 4, 16
        else:
            bs, acc = 2, 32
    elif tag == "huge_v2":
        if h <= 128:
            bs, acc = 4, 16
        elif h <= 256:
            bs, acc = 2, 32
        else:
            bs, acc = 1, 64
    elif tag in ["small_ft_kin", "base_ft_kin", "small_ssl", "base_ssl"]:
        if h <= 256:
            bs, acc = 32, 4
        else:
            bs, acc = 16, 8
    elif tag in ["large_ft_kin", "large_ssl"]:
        if h <= 128:
            bs, acc = 16, 8
        elif h <= 256:
            bs, acc = 8, 16
        else:
            bs, acc = 4, 32
    elif tag in ["huge_ft_kin", "huge_ssl"]:
        if h <= 128:
            bs, acc = 8, 16
        elif h <= 256:
            bs, acc = 4, 32
        else:
            bs, acc = 2, 64
    else:
        bs, acc = 8, 16

    eff = bs * acc
    if eff < TOKEN_PARITY_EFFECTIVE_BATCH:
        acc = math.ceil(TOKEN_PARITY_EFFECTIVE_BATCH / bs)
    return bs, acc

def manifest_has_run(manifest_path, run_name):
    if not os.path.exists(manifest_path):
        return False
    with open(manifest_path, "r") as f:
        for line in f:
            if f'"run_name": "{run_name}"' in line:
                return True
    return False

def design_key_str(protocol, tag, hidden, strategy, layers, unfreeze):
    return f"{protocol}|backbone={tag}|hidden={hidden}|video_strategy={strategy}|video_layers={layers}|embed={unfreeze}"

def run_name_str(tag, hidden, seed, strategy, layers, ladder_suffix: str = ""):
    strat_suffix = f"{strategy}{layers}" if strategy == "last_n" else strategy
    base = f"S_{tag}_{strat_suffix}_h{hidden}_s{seed}"
    return f"{base}{ladder_suffix}" if ladder_suffix else base


def ladder_suffix_for(mult: float) -> str:
    label = f"{mult:g}".replace(".", "p")
    return f"_ladder{label}"


MATRIX_PRESETS: dict[str, dict[str, str]] = {
    # Paper-primary: full 3x3 backbone/hidden matrix with two seeds (18 runs).
    "primary": {
        "SCALING_PROTOCOL_NAME": "paper_scaling_full_matrix_2seed_v1",
        "SCALING_COMPUTE_MODE": "token_parity",
        "SCALING_HIDDENS": "128 256 512",
        "SCALING_SEEDS": "0 1",
        "SCALING_REF_VIDEO_MODEL_NAME": "OpenGVLab/VideoMAEv2-Large",
        "SCALING_REF_HIDDEN": "256",
        "SCALING_MAX_EPOCHS_REF": "30",
        "SCALING_STEP_MIN": "0",
        "SCALING_VIDEO_TRAINABLE_LAYERS": "2",
        "SCALING_COMPILE": "0",
        "SCALING_EARLY_STOPPING_PATIENCE": "null",
        "SCALING_CHECK_VAL_EVERY_N_EPOCH": "1",
        "SCALING_VAL_CHECK_INTERVAL": "500",
        "SCALING_LIMIT_VAL_BATCHES": "0.25",
        "SCALING_PROGRESS_EVERY_N_BATCHES": "100",
        "EVAL_TWENTE": "0",
    },
    # Fastest sanity path: base+large, two seeds, no Twente (~4 runs).
    "minimal": {
        "SCALING_PROTOCOL_NAME": "paper_scaling_token_parity_flop_reported_v1",
        "SCALING_COMPUTE_MODE": "token_parity",
        "SCALING_BACKBONES": (
            "base_v2|OpenGVLab/VideoMAEv2-Base\n"
            "large_v2|OpenGVLab/VideoMAEv2-Large"
        ),
        "SCALING_HIDDENS": "256",
        "SCALING_SEEDS": "0 1",
        "SCALING_REF_VIDEO_MODEL_NAME": "OpenGVLab/VideoMAEv2-Large",
        "SCALING_REF_HIDDEN": "256",
        "SCALING_MAX_EPOCHS_REF": "50",
        "SCALING_STEP_MIN": "0",
        "SCALING_VIDEO_TRAINABLE_LAYERS": "2",
        "SCALING_COMPILE": "1",
        "SCALING_EARLY_STOPPING_PATIENCE": "null",
        "EVAL_TWENTE": "0",
    },
    # Full hidden sweep (27 runs) — only when you explicitly opt in.
    "full": {
        "SCALING_PROTOCOL_NAME": "paper_scaling_token_parity_flop_reported_v1",
        "SCALING_COMPUTE_MODE": "token_parity",
        "SCALING_HIDDENS": "128 256 512",
        "SCALING_SEEDS": "0 1 2",
        "SCALING_REF_VIDEO_MODEL_NAME": "OpenGVLab/VideoMAEv2-Large",
        "SCALING_REF_HIDDEN": "256",
        "SCALING_MAX_EPOCHS_REF": "100",
        "SCALING_STEP_MIN": "0",
        "SCALING_VIDEO_TRAINABLE_LAYERS": "2",
        "SCALING_COMPILE": "1",
        "SCALING_EARLY_STOPPING_PATIENCE": "null",
        "EVAL_TWENTE": "1",
    },
}


def apply_scaling_matrix_preset(matrix: str) -> None:
    preset = MATRIX_PRESETS.get(matrix)
    if preset is None:
        print(f"ERROR: unknown SCALING_MATRIX={matrix!r}. Choose: {', '.join(MATRIX_PRESETS)}", file=sys.stderr)
        sys.exit(1)
    for key, value in preset.items():
        os.environ.setdefault(key, value)


def scaling_design_enabled(matrix: str, tag: str, hidden: str, primary_hidden: str, hidden_sweep_backbone: str) -> bool:
    if matrix != "primary" or not hidden_sweep_backbone:
        return True
    return str(hidden) == str(primary_hidden) or tag == hidden_sweep_backbone


def autofill_full_cache_paths() -> None:
    train_cache = ROOT / "data/processed/full_cache/train"
    val_cache = ROOT / "data/processed/full_cache/val"
    if train_cache.is_dir():
        os.environ.setdefault("SCALING_TRAIN_FULL_CACHE", str(train_cache))
    if val_cache.is_dir():
        os.environ.setdefault("SCALING_VAL_FULL_CACHE", str(val_cache))


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root)
    import argparse
    import random

    parser = argparse.ArgumentParser()
    parser.add_argument("--random-seeds", action="store_true", help="Replace explicit SCALING_SEEDS with 30 random seeds")
    args, _ = parser.parse_known_args()
    
    if get_env("PILOT", "0") != "1":
        apply_scaling_matrix_preset(get_env("SCALING_MATRIX", "primary"))
        autofill_full_cache_paths()

    config = get_env("CONFIG", "egomuscle/training/config.yaml")
    manifest = get_env("MANIFEST", "experiments/results/scaling_manifest.jsonl")
    pilot = get_env("PILOT", "0")
    eval_twente = get_env("EVAL_TWENTE", "0")
    scaling_matrix = get_env("SCALING_MATRIX", "primary")

    scaling_protocol_name = get_env("SCALING_PROTOCOL_NAME", "paper_scaling_token_parity_flop_reported_v1")
    scaling_seeds = get_env("SCALING_SEEDS", "0 1 2").split()
    if args.random_seeds:
        scaling_seeds = [str(random.randint(0, 2**31 - 1)) for _ in range(30)]
    scaling_hiddens = get_env("SCALING_HIDDENS", "256").split()
    scaling_primary_hidden = get_env("SCALING_PRIMARY_HIDDEN", "256")
    scaling_hidden_sweep_backbone = get_env("SCALING_HIDDEN_SWEEP_BACKBONE", "")
    scaling_video_trainable_strategy = get_env("SCALING_VIDEO_TRAINABLE_STRATEGY", "last_n")
    scaling_video_trainable_layers = get_env("SCALING_VIDEO_TRAINABLE_LAYERS", "2")
    scaling_video_unfreeze_embeddings = get_env("SCALING_VIDEO_UNFREEZE_EMBEDDINGS", "0")
    scaling_skip_existing = get_env("SCALING_SKIP_EXISTING", "1")
    scaling_compute_mode = get_env("SCALING_COMPUTE_MODE", "token_parity")
    scaling_allow_budget_clamp = get_env("SCALING_ALLOW_BUDGET_CLAMP", "0")
    scaling_step_min = get_env("SCALING_STEP_MIN", "")
    if not scaling_step_min:
        scaling_step_min = "0" if scaling_compute_mode == "token_parity" else "400"
    scaling_warmup_ratio = get_env("SCALING_WARMUP_RATIO", "0.05")
    scaling_warmup_steps = get_env("SCALING_WARMUP_STEPS", "")
    scaling_warmup_steps_cap = get_env("SCALING_WARMUP_STEPS_CAP", "100")
    scaling_compute_ladder = get_env("SCALING_COMPUTE_LADDER", "")
    scaling_ladder_backbone = get_env("SCALING_LADDER_BACKBONE", "large_v2")
    scaling_ladder_hidden = get_env("SCALING_LADDER_HIDDEN", "256")
    scaling_step_max = get_env("SCALING_STEP_MAX", "")
    scaling_use_wandb = get_env("SCALING_USE_WANDB", "0")
    scaling_wandb_project = get_env("SCALING_WANDB_PROJECT", "egomuscle")
    scaling_wandb_entity = get_env("SCALING_WANDB_ENTITY", "")
    scaling_wandb_mode = get_env("SCALING_WANDB_MODE", "online")
    scaling_wandb_group = get_env("SCALING_WANDB_GROUP", "scaling_law_v2")
    scaling_virtual_sampling = get_env("SCALING_VIRTUAL_SAMPLING", "1")
    scaling_dry_run = get_env("SCALING_DRY_RUN", "0")
    scaling_check_val_every_n_epoch = get_env("SCALING_CHECK_VAL_EVERY_N_EPOCH", "")
    scaling_val_check_interval = get_env("SCALING_VAL_CHECK_INTERVAL", "")
    scaling_limit_val_batches = get_env("SCALING_LIMIT_VAL_BATCHES", "")
    scaling_progress_every_n_batches = get_env("SCALING_PROGRESS_EVERY_N_BATCHES", "")

    os.makedirs(os.path.dirname(manifest) or ".", exist_ok=True)

    backbones_str = get_env("SCALING_BACKBONES", "base_v2|OpenGVLab/VideoMAEv2-Base\nlarge_v2|OpenGVLab/VideoMAEv2-Large\nhuge_v2|OpenGVLab/VideoMAEv2-Huge")
    backbones = []
    for line in backbones_str.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"): continue
        parts = line.split("|")
        if len(parts) >= 2:
            backbones.append((parts[0].strip(), parts[1].strip()))

    max_epoch_ref_str = get_env("SCALING_MAX_EPOCHS_REF", "")
    if not max_epoch_ref_str:
        max_epoch_ref = read_training_max_epochs(config)
    else:
        max_epoch_ref = int(max_epoch_ref_str)
    budget_fraction = float(get_env("SCALING_BUDGET_FRACTION", "1.0"))
    if budget_fraction > 0 and abs(budget_fraction - 1.0) > 1e-9:
        max_epoch_ref = max(1, int(round(max_epoch_ref * budget_fraction)))

    ref_tag, ref_vid_raw = backbones[0] if backbones else ("", "")
    ref_vid_name = get_env("SCALING_REF_VIDEO_MODEL_NAME", ref_vid_raw)
    ref_hidden = get_env("SCALING_REF_HIDDEN", scaling_hiddens[0] if scaling_hiddens else "")
    ref_batch = get_env("SCALING_REF_BATCH", "")
    ref_accum = get_env("SCALING_REF_ACCUM", "")
    
    ref_vid_name = resolve_video_model_id("ref", ref_vid_name)
    validate_video_model_id("ref", ref_vid_name)

    ladder_run_mult = 1
    if scaling_compute_ladder.strip() and scaling_ladder_backbone and scaling_ladder_hidden in scaling_hiddens:
        ladder_run_mult = len([x for x in scaling_compute_ladder.split(",") if x.strip()])

    planned_designs = [
        (tag, hidden)
        for tag, _vid in backbones
        for hidden in scaling_hiddens
        if scaling_design_enabled(
            scaling_matrix,
            tag,
            hidden,
            scaling_primary_hidden,
            scaling_hidden_sweep_backbone,
        )
    ]
    planned_runs = len(scaling_seeds) * len(planned_designs) * ladder_run_mult
    print(
        f"=== scaling sweep plan: matrix={scaling_matrix} runs={planned_runs} "
        f"backbones={len(backbones)} hiddens={scaling_hiddens} seeds={scaling_seeds} "
        f"primary_hidden={scaling_primary_hidden} hidden_sweep_backbone={scaling_hidden_sweep_backbone or 'all'} "
        f"max_epochs_ref={max_epoch_ref} eval_twente={eval_twente} manifest={manifest} ===",
        flush=True,
    )
    if scaling_matrix != "full" and len(scaling_hiddens) > 1 and not scaling_hidden_sweep_backbone:
        print(
            "NOTE: multiple hiddens scheduled; for the 9-run primary matrix use SCALING_MATRIX=primary "
            "(default) or SCALING_HIDDENS='256'. Full 27-run sweep: SCALING_MATRIX=full.",
            flush=True,
        )

    for seed in scaling_seeds:
        for tag, vid_raw in backbones:
            vid = resolve_video_model_id(tag, vid_raw)
            validate_video_model_id(tag, vid)
            for hidden in scaling_hiddens:
                if not scaling_design_enabled(
                    scaling_matrix,
                    tag,
                    hidden,
                    scaling_primary_hidden,
                    scaling_hidden_sweep_backbone,
                ):
                    continue
                ladder_mults = [1.0]
                if scaling_compute_ladder.strip():
                    if tag == scaling_ladder_backbone and str(hidden) == str(scaling_ladder_hidden):
                        ladder_mults = [float(x.strip()) for x in scaling_compute_ladder.split(",") if x.strip()]

                for ladder_mult in ladder_mults:
                    lsuffix = "" if abs(ladder_mult - 1.0) < 1e-9 else ladder_suffix_for(ladder_mult)
                    run = run_name_str(
                        tag, hidden, seed, scaling_video_trainable_strategy, scaling_video_trainable_layers, lsuffix
                    )
                    design_key = design_key_str(
                        scaling_protocol_name, tag, hidden, scaling_video_trainable_strategy,
                        scaling_video_trainable_layers, scaling_video_unfreeze_embeddings,
                    )
                    if lsuffix:
                        design_key = f"{design_key}|ladder={ladder_mult:g}"

                    if scaling_skip_existing == "1" and manifest_has_run(manifest, run):
                        print(f"=== scaling sweep: skip existing run_name={run} ===")
                        continue

                    bs, acc = vram_batch_accum(tag, hidden)

                    schedule_env = {}
                    if pilot == "1":
                        schedule_env.update({
                            "max_steps": "8", "max_epochs_cap": "1", "mode": "fixed", "train_samples": "0", "device_count": "1",
                            "micro_batches_per_epoch": "0", "optimizer_steps_per_epoch": "1", "video_tokens_per_sample": "0",
                            "effective_batch_size": str(bs * acc), "video_tokens_per_optimizer_step": "0", "total_forward_flops_per_sample": "0",
                            "frozen_forward_flops_per_sample": "0", "trainable_forward_flops_per_sample": "0", "train_flops_per_optimizer_step": "0",
                            "ref_video_model_name": ref_vid_name, "ref_muscle_hidden_dim": str(ref_hidden), "ref_effective_batch_size": "0",
                            "ref_optimizer_steps_per_epoch": "0", "ref_video_tokens_per_optimizer_step": "0", "ref_train_flops_per_optimizer_step": "0",
                            "reference_total_video_tokens": "0", "reference_total_train_flops": "0", "unclamped_max_steps": "8",
                            "rounded_max_steps": "8", "step_min": "0", "step_max": "8", "budget_was_clamped": "False", "budget_clamp_reason": ""
                        })
                    else:
                        sched_args = [
                            "python", "experiments/scaling_compute.py", "schedule",
                            "--config", config, "--video-model-name", vid, "--muscle-hidden-dim", str(hidden),
                            "--batch-size", str(bs), "--accumulate-grad-batches", str(acc),
                            "--mode", scaling_compute_mode, "--baseline-epochs", str(max_epoch_ref),
                            "--epoch-min", get_env("SCALING_EPOCH_MIN", "0"),
                            "--epoch-max", get_env("SCALING_EPOCH_MAX", "200"),
                            "--video-trainable-strategy", scaling_video_trainable_strategy,
                            "--video-trainable-layers", scaling_video_trainable_layers,
                            "--video-unfreeze-embeddings", scaling_video_unfreeze_embeddings,
                            "--format", "shell",
                        ]
                        if scaling_step_min:
                            sched_args.extend(["--step-min", scaling_step_min])
                        if scaling_step_max:
                            sched_args.extend(["--step-max", scaling_step_max])
                        if ref_vid_name:
                            sched_args.extend(["--ref-video-model-name", ref_vid_name])
                        if ref_hidden:
                            sched_args.extend(["--ref-muscle-hidden-dim", str(ref_hidden)])
                        if ref_batch:
                            sched_args.extend(["--ref-batch-size", str(ref_batch)])
                        if ref_accum:
                            sched_args.extend(["--ref-accumulate-grad-batches", str(ref_accum)])

                        try:
                            proc = subprocess.run(sched_args, capture_output=True, text=True, check=True, cwd=ROOT)
                        except subprocess.CalledProcessError as exc:
                            if exc.stderr:
                                print(exc.stderr, file=sys.stderr)
                            raise
                        for line in proc.stdout.split("\n"):
                            if "=" in line:
                                k, v = line.split("=", 1)
                                schedule_env[k.strip()] = v.strip().strip("'\"")

                        if schedule_env.get("mode") != "fixed" and schedule_env.get("budget_was_clamped") == "True" and scaling_allow_budget_clamp != "1":
                            if not (schedule_env.get("budget_clamp_reason") == "step_min" and scaling_step_min):
                                print(f"ERROR: parity budget for {run} was clamped from {schedule_env.get('unclamped_max_steps')} to {schedule_env.get('max_steps')}.", file=sys.stderr)
                                sys.exit(1)

                    base_max_steps = int(schedule_env.get("max_steps", "0") or "0")
                    if abs(ladder_mult - 1.0) >= 1e-9:
                        max_steps_int = max(1, int(round(base_max_steps * ladder_mult)))
                        steps_per_epoch = int(schedule_env.get("optimizer_steps_per_epoch", "1") or "1")
                        schedule_env["max_steps"] = str(max_steps_int)
                        schedule_env["max_epochs_cap"] = str(max(1, math.ceil(max_steps_int / max(1, steps_per_epoch)) + 2))
                    else:
                        max_steps_int = base_max_steps

                    print(
                        f"=== scaling sweep: protocol={scaling_protocol_name} design_key={design_key} "
                        f"run_name={run} batch={bs} accumulate={acc} max_steps={schedule_env.get('max_steps')} "
                        f"max_epochs_cap={schedule_env.get('max_epochs_cap')} ladder_mult={ladder_mult:g} ==="
                    )

                    use_wandb = "true" if scaling_use_wandb == "1" else "false"
                    unfreeze_bool = "true" if scaling_video_unfreeze_embeddings == "1" else "false"
                    effective_batch_int = int(schedule_env.get("effective_batch_size", str(bs * acc)) or str(bs * acc))
                    train_samples_int = int(schedule_env.get("train_samples", "0") or "0")
                    consumed_train_examples = max_steps_int * effective_batch_int
                    exposure_multiplier = (
                        float(consumed_train_examples) / float(train_samples_int)
                        if train_samples_int > 0
                        else 0.0
                    )
                    virtual_size = max(train_samples_int, consumed_train_examples) if scaling_virtual_sampling == "1" else 0

                    train_args = [
                        "python", "-m", "egomuscle.training.train",
                        "--config", config,
                        "--override", f"seed={seed}",
                        "--override", f"logging.run_name={run}",
                        "--override", f"logging.group={scaling_wandb_group}",
                        "--override", "logging.tags=[scaling_law,backbone_trainable]",
                        "--override", f"logging.use_wandb={use_wandb}",
                        "--override", f"logging.project={scaling_wandb_project}",
                        "--override", f"logging.wandb_mode={scaling_wandb_mode}",
                        "--override", f"model.video_model_name={vid}",
                        "--override", f"model.video_trainable_strategy={scaling_video_trainable_strategy}",
                        "--override", f"model.video_trainable_layers={scaling_video_trainable_layers}",
                        "--override", f"model.video_unfreeze_embeddings={unfreeze_bool}",
                        "--override", f"model.muscle_hidden_dim={hidden}",
                        "--override", "model.fusion_mode=cross_attn",
                        "--override", "model.use_video=true",
                        "--override", "model.use_muscle=true",
                        "--override", "model.label_conditioning=false",
                        "--override", f"model.fusion_dropout={get_env('SCALING_FUSION_DROPOUT', '0.1')}",
                        "--override", f"model.pred_dropout={get_env('SCALING_PRED_DROPOUT', '0.1')}",
                        "--override", f"data.batch_size={bs}",
                        "--override", f"data.num_workers={get_env('SCALING_NUM_WORKERS', '12')}",
                        "--override", f"data.train.full_cache_dir={get_env('SCALING_TRAIN_FULL_CACHE', '')}",
                        "--override", f"data.val.full_cache_dir={get_env('SCALING_VAL_FULL_CACHE', '')}",
                        "--override", f"data.temporal_sample_mode={get_env('SCALING_TEMPORAL_SAMPLE_MODE', 'random_stride')}",
                        "--override", f"data.train.muscle_noise_std={get_env('SCALING_MUSCLE_NOISE_STD', '0.01')}",
                        "--override", f"data.train.replacement_sampling={'true' if scaling_virtual_sampling == '1' else 'false'}",
                        "--override", f"data.train.virtual_size={virtual_size if virtual_size > 0 else 'null'}",
                        "--override", f"training.accumulate_grad_batches={acc}",
                        "--override", f"training.max_steps={schedule_env.get('max_steps')}",
                        "--override", f"training.max_epochs={schedule_env.get('max_epochs_cap')}",
                        "--override", f"training.compile={'true' if get_env('SCALING_COMPILE', '1') == '1' else 'false'}",
                        "--override", f"training.compile_mode={get_env('SCALING_COMPILE_MODE', 'default')}",
                        "--override", f"training.warmup_ratio={scaling_warmup_ratio}",
                        "--override", f"training.warmup_steps_cap={scaling_warmup_steps_cap}",
                        "--override", f"training.early_stopping_patience={get_env('SCALING_EARLY_STOPPING_PATIENCE', 'null' if scaling_compute_mode == 'token_parity' else '15')}",
                        "--override", f"scaling.protocol_name={scaling_protocol_name}",
                        "--override", f"scaling.design_key={design_key}",
                        "--override", f"scaling.seed={seed}",
                        "--override", f"scaling.backbone_tag={tag}",
                        "--override", f"scaling.compute_mode={schedule_env.get('mode')}",
                        "--override", f"scaling.train_samples={schedule_env.get('train_samples')}",
                        "--override", f"scaling.device_count={schedule_env.get('device_count')}",
                        "--override", f"scaling.micro_batches_per_epoch={schedule_env.get('micro_batches_per_epoch')}",
                        "--override", f"scaling.optimizer_steps_per_epoch={schedule_env.get('optimizer_steps_per_epoch')}",
                        "--override", f"scaling.video_tokens_per_sample={schedule_env.get('video_tokens_per_sample')}",
                        "--override", f"scaling.effective_batch_size={schedule_env.get('effective_batch_size')}",
                        "--override", f"scaling.video_tokens_per_optimizer_step={schedule_env.get('video_tokens_per_optimizer_step')}",
                        "--override", f"scaling.total_forward_flops_per_sample={schedule_env.get('total_forward_flops_per_sample')}",
                        "--override", f"scaling.frozen_forward_flops_per_sample={schedule_env.get('frozen_forward_flops_per_sample')}",
                        "--override", f"scaling.trainable_forward_flops_per_sample={schedule_env.get('trainable_forward_flops_per_sample')}",
                        "--override", f"scaling.train_flops_per_optimizer_step={schedule_env.get('train_flops_per_optimizer_step')}",
                        "--override", f"scaling.virtual_sampling={scaling_virtual_sampling}",
                        "--override", f"scaling.consumed_train_examples={consumed_train_examples}",
                        "--override", f"scaling.exposure_multiplier={exposure_multiplier}"
                    ]

                    lr_base = float(get_env("SCALING_LEARNING_RATE_BASE", "3e-4"))
                    lr_mode = get_env("SCALING_LR_MODE", "constant")
                
                    if lr_mode == "constant":
                        lr = lr_base
                    elif lr_mode == "backbone_scaled":
                        if tag == "base_ft_kin": lr = lr_base * 0.7
                        elif tag == "large_ft_kin": lr = lr_base * 0.5
                        elif tag == "huge_ft_kin": lr = lr_base * 0.3
                        else: lr = lr_base
                    else:
                        print(f"ERROR: unsupported SCALING_LR_MODE={lr_mode}", file=sys.stderr)
                        sys.exit(1)
                    
                    train_args.extend([
                        "--override", f"training.learning_rate={lr}",
                        "--override", f"scaling.learning_rate={lr}",
                        "--override", f"scaling.lr_mode={lr_mode}"
                    ])
                    if scaling_warmup_steps:
                        train_args.extend(["--override", f"training.warmup_steps={scaling_warmup_steps}"])
                    if scaling_check_val_every_n_epoch:
                        train_args.extend(["--override", f"trainer.check_val_every_n_epoch={scaling_check_val_every_n_epoch}"])
                    if scaling_val_check_interval:
                        train_args.extend(["--override", f"trainer.val_check_interval={scaling_val_check_interval}"])
                    if scaling_limit_val_batches:
                        train_args.extend(["--override", f"trainer.limit_val_batches={scaling_limit_val_batches}"])
                    if scaling_progress_every_n_batches:
                        train_args.extend(["--override", f"trainer.progress_every_n_batches={scaling_progress_every_n_batches}"])
                    if scaling_wandb_entity:
                        train_args.extend(["--override", f"logging.entity={scaling_wandb_entity}"])
                    if lsuffix:
                        train_args.extend(["--override", f"scaling.ladder_mult={ladder_mult:g}"])

                    if scaling_dry_run == "1":
                        print("DRY_RUN train command:")
                        print(" ".join(train_args))
                        continue
                    
                    subprocess.run(train_args, check=True)

                    twente_json = ""
                    if eval_twente == "1":
                        twente_json = f"experiments/results/twente_{run}.json"
                        ckpt_dir = Path(f"checkpoints/{run}")
                        ckpts = sorted(ckpt_dir.glob("*.ckpt")) if ckpt_dir.exists() else []
                        if ckpts:
                            subprocess.run([
                                "python", "-m", "egomuscle.eval.twente_eval",
                                "--checkpoint", str(ckpts[-1]),
                                "--config", config,
                                "--output", twente_json,
                                "--override", f"model.video_model_name={vid}",
                                "--override", f"model.video_trainable_strategy={scaling_video_trainable_strategy}",
                                "--override", f"model.video_trainable_layers={scaling_video_trainable_layers}",
                                "--override", f"model.video_unfreeze_embeddings={unfreeze_bool}",
                                "--override", f"model.muscle_hidden_dim={hidden}",
                                "--override", "model.fusion_mode=cross_attn",
                                "--override", "model.use_video=true",
                                "--override", "model.use_muscle=true",
                                "--override", "model.label_conditioning=false"
                            ], check=True)
                        else:
                            print(f"WARN: no checkpoint under checkpoints/{run}; skip Twente eval", file=sys.stderr)

                    append_args = [
                        "python", "experiments/fit_scaling_law.py", "append",
                        "--run-name", run, "--protocol-name", scaling_protocol_name,
                        "--design-key", design_key, "--seed", str(seed),
                        "--backbone-tag", tag, "--manifest", manifest,
                        "--log-root", "lightning_logs", "--video-model-name", vid,
                        "--muscle-hidden-dim", str(hidden), "--video-trainable-strategy", scaling_video_trainable_strategy,
                        "--video-trainable-layers", scaling_video_trainable_layers, "--video-unfreeze-embeddings", scaling_video_unfreeze_embeddings,
                        "--batch-size", str(bs), "--accumulate-grad-batches", str(acc),
                        "--fusion-mode", "cross_attn", "--training-max-epochs", schedule_env.get('max_epochs_cap', ''),
                        "--training-max-steps", schedule_env.get('max_steps', ''), "--compute-mode", schedule_env.get('mode', ''),
                        "--train-samples", schedule_env.get('train_samples', ''), "--device-count", schedule_env.get('device_count', ''),
                        "--micro-batches-per-epoch", schedule_env.get('micro_batches_per_epoch', ''), "--optimizer-steps-per-epoch", schedule_env.get('optimizer_steps_per_epoch', ''),
                        "--video-tokens-per-sample", schedule_env.get('video_tokens_per_sample', ''), "--effective-batch-size", schedule_env.get('effective_batch_size', ''),
                        "--video-tokens-per-optimizer-step", schedule_env.get('video_tokens_per_optimizer_step', ''), "--total-forward-flops-per-sample", schedule_env.get('total_forward_flops_per_sample', ''),
                        "--frozen-forward-flops-per-sample", schedule_env.get('frozen_forward_flops_per_sample', ''), "--trainable-forward-flops-per-sample", schedule_env.get('trainable_forward_flops_per_sample', ''),
                        "--train-flops-per-optimizer-step", schedule_env.get('train_flops_per_optimizer_step', ''), "--epoch-baseline", str(max_epoch_ref),
                        "--virtual-sampling", scaling_virtual_sampling, "--consumed-train-examples", str(consumed_train_examples), "--exposure-multiplier", str(exposure_multiplier),
                        "--use-wandb", scaling_use_wandb, "--wandb-project", scaling_wandb_project, "--wandb-mode", scaling_wandb_mode
                    ]
                
                    for k, v in [
                        ("--ref-video-model-name", ref_vid_name), ("--ref-muscle-hidden-dim", str(ref_hidden)),
                        ("--ref-effective-batch-size", schedule_env.get('ref_effective_batch_size')), ("--ref-optimizer-steps-per-epoch", schedule_env.get('ref_optimizer_steps_per_epoch')),
                        ("--ref-video-tokens-per-optimizer-step", schedule_env.get('ref_video_tokens_per_optimizer_step')), ("--ref-train-flops-per-optimizer-step", schedule_env.get('ref_train_flops_per_optimizer_step')),
                        ("--reference-total-video-tokens", schedule_env.get('reference_total_video_tokens')), ("--reference-total-train-flops", schedule_env.get('reference_total_train_flops')),
                        ("--unclamped-max-steps", schedule_env.get('unclamped_max_steps')), ("--rounded-max-steps", schedule_env.get('rounded_max_steps')),
                        ("--step-min", schedule_env.get('step_min')), ("--step-max", schedule_env.get('step_max')),
                        ("--budget-clamp-reason", schedule_env.get('budget_clamp_reason')), ("--wandb-entity", scaling_wandb_entity),
                        ("--wandb-group", scaling_wandb_group)
                    ]:
                        if v: append_args.extend([k, v])
                    
                    append_args.extend(["--budget-was-clamped", "1" if schedule_env.get('budget_was_clamped') == "True" else "0"])
                    if lsuffix:
                        append_args.extend(["--ladder-mult", str(ladder_mult)])
                    if twente_json and os.path.exists(twente_json):
                        append_args.extend(["--twente-json", twente_json])

                    subprocess.run(append_args, check=True)

    print(f"Done. Manifest: {manifest}")
    print("Fit examples (Analysis A, h=256):")
    manifest_q = manifest
    print(f"  A1: python experiments/fit_scaling_law.py fit --manifest {manifest_q} --n-key model/trainable_params --aggregate-by design_key --filter-hidden 256")
    print(f"  A2: python experiments/fit_scaling_law.py fit --manifest {manifest_q} --n-key compute/total_train_flops --aggregate-by design_key --filter-hidden 256")
    print(f"  A3: python experiments/fit_scaling_law.py fit --manifest {manifest_q} --n-key compute/flop_adjusted_trainable_params --aggregate-by design_key --filter-hidden 256")

if __name__ == "__main__":
    main()
