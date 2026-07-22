# EgoMuscle

EgoMuscle is a research codebase for training video + muscle-activation models and evaluating their representations with RSA against human fMRI-derived RDMs.

## What is in this repo

| Path | Purpose |
| --- | --- |
| `egomuscle/` | Python package for data preparation, models, training, and evaluation. |
| `experiments/` | Maintained experiment entry points and small experiment-local inputs. |
| `scripts/downloaders/` | Download helpers for public assets. |
| `metadata.json` | Tracked Twente split metadata. Keep this file in git. |

Generated datasets, checkpoints, caches, W&B runs, and `experiments/results/` outputs should stay out of git.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

Or run the bootstrap helper:

```bash
bash scripts/setup_egomuscle.sh
source .venv/bin/activate
source scripts/activate_egomuscle.sh
```

Download public helper assets with:

```bash
bash scripts/downloaders/download_egomuscle_data.sh
```

This covers VideoMAE weights, MinT, and Twente. AMASS, BABEL, SMPL+H, and Algonauts 2025 require manual access from their official sources.

## Maintained commands

| Task | Command |
| --- | --- |
| Train with defaults | `python -m egomuscle.training.train --config egomuscle/training/config.yaml` |
| Full clean run | `python experiments/run_clean_full_experiments.py` |
| Scaling sweep | `python experiments/run_scaling_sweep.py` |
| Ablations | `python experiments/run_ablations_csv.py` |
| BABEL recentered/raw view pair | `python experiments/build_babel_viewpair.py` |
| PBIT quantization sweep | `python experiments/run_pbit_quantization_sweep.py` |
| Algonauts 2025 RSA benchmark | `python experiments/run_algonauts2025_rsa.py --checkpoint <checkpoint>` |
| Layerwise neural RSA | `python experiments/run_layerwise_hierarchy.py --checkpoint <checkpoint>` |
| Viability-boundary RSA | `python experiments/run_viability_boundary_rsa.py` |

Pass config overrides with dotted keys, for example:

```bash
python -m egomuscle.training.train --config egomuscle/training/config.yaml \
  --override training.max_epochs=10
```

## Required data layout

### AMASS

Download AMASS from its official site and place `.npz` sequences under:

```text
data/raw/amass/
```

The current AMASS-rendered pipeline expects BMLmovi, BMLrub, EyesJapanDataset, KIT, and TotalCapture with paths matching the MinT/BABEL references.

### BABEL

Place BABEL split JSON files under:

```text
data/raw/babel/babel_v1.0_release/train.json
data/raw/babel/babel_v1.0_release/val.json
data/raw/babel/babel_v1.0_release/test.json
```

Include `extra_train.json` and `extra_val.json` when available.

### SMPL+H

Place body models under:

```text
data/raw/models/smplh/SMPLH_FEMALE.pkl
data/raw/models/smplh/SMPLH_MALE.pkl
data/raw/models/smplh/SMPLH_NEUTRAL.pkl
```

Set `SMPL_MODEL_DIR=data/raw/models` when using a different root.

## Build processed clips

With MinT, BABEL, AMASS, and SMPL+H available:

```bash
python -m egomuscle.data.build_manifests \
  --mint-root data/raw/mint_extracted \
  --babel-root data/raw/babel \
  --output-dir data/processed/manifests

AMASS_ROOT=data/raw/amass SMPL_ROOT=data/raw/models \
  python experiments/build_amass_pair.py data/processed data/processed_exo
```

The setup script can run the same build path:

```bash
FULL_BUILD=1 bash scripts/setup_egomuscle.sh
```

For a small build:

```bash
MAX_RECORDS=20 FULL_BUILD=1 bash scripts/setup_egomuscle.sh
```

## Twente real EMG

Download and extract Twente with:

```bash
bash scripts/downloaders/download_twente.sh
```

Then build the eval layout:

```bash
python -m egomuscle.data.build_manifests \
  --twente-emg-root data/raw/twente_emg/extracted \
  --twente-rgb-root data/raw/twente_rgb/extracted \
  --output-dir data/processed/manifests
python -m egomuscle.data.build_twente_dataset --view front
```

Eval defaults to `data/processed_real/twente`.

## Algonauts 2025 neural RSA

Download the dataset with DataLad (or curl fallback):

```bash
bash scripts/downloaders/download_algonauts2025.sh
```

Expected layout after download:

```text
data/raw/algonauts2025/
├── fmri/                           # Subject-level .h5 files
└── stimuli/movies/                 # Movie10 MKV files
```

Build neural RDMs from fMRI:

```bash
python -m egomuscle.eval.prepare_algonauts2025_rdms \
  --fmri-root data/raw/algonauts2025/fmri \
  --output-dir egomuscle/eval/algonauts2025_rdms
```

Build the clip manifest (TR-to-frame mapping):

```bash
python experiments/build_algonauts2025_clips.py \
  --h5 data/raw/algonauts2025/fmri/sub-01/func/sub-01_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5 \
  --stimuli-root data/raw/algonauts2025/stimuli/movies/movie10 \
  --output experiments/results/algonauts2025_clip_manifest.jsonl
```

Run the full RSA benchmark (feature extraction, RDMs, scoring, permutation tests):

```bash
python experiments/run_algonauts2025_rsa.py \
  --checkpoint <checkpoint.ckpt> \
  --manifest experiments/results/algonauts2025_clip_manifest.jsonl \
  --stimuli-root data/raw/algonauts2025/stimuli/movies/movie10 \
  --neural-dir egomuscle/eval/algonauts2025_rdms \
  --output-dir experiments/results/algonauts2025_rsa
```

During training, the validation loop also monitors Algonauts RSA for `bourne01`/`bourne02` movies (Visual and DefaultMode networks) when `evaluation.algonauts2025_rdms_dir` is configured. See `egomuscle/training/config.yaml`.

