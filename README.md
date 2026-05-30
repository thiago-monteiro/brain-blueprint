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

This covers VideoMAE weights, MinT, and Twente. AMASS, BABEL, SMPL+H, and BOLD5000 stimuli require manual access from their official sources.

## Maintained commands

| Task | Command |
| --- | --- |
| Train with defaults | `python -m egomuscle.training.train --config egomuscle/training/config.yaml` |
| Full clean run | `python experiments/run_clean_full_experiments.py` |
| Scaling sweep | `python experiments/run_scaling_sweep.py` |
| Ablations | `python experiments/run_ablations_csv.py` |
| BABEL recentered/raw view pair | `python experiments/build_babel_viewpair.py` |
| SMFE train/eval suite | `python experiments/run_smfe_suite.py --mode train` or `--mode eval` |
| PBIT quantization sweep | `python experiments/run_pbit_quantization_sweep.py` |
| Layerwise neural RSA | `python experiments/run_layerwise_hierarchy.py` |
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

## BOLD5000 neural RSA

Expected local layout:

```text
data/raw/bold5000/extracted/BOLD5000_GLMsingle_ROI_betas/mat/
data/raw/bold5000/extracted/BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli/
data/raw/bold5000/extracted/BOLD5000_Stimuli/Stimuli_Presentation_Lists/
```

Download ROI betas:

```bash
bash scripts/downloaders/download_bold5000_roi_betas.sh
unzip data/raw/bold5000/BOLD5000_GLMsingle_ROI_betas.zip -d data/raw/bold5000/extracted
```

Build neural RDMs:

```bash
python -m egomuscle.eval.prepare_bold5000_rdms \
  --mat-root data/raw/bold5000/extracted/BOLD5000_GLMsingle_ROI_betas/mat \
  --output-dir egomuscle/eval/neural_rdms
```

Build the stimulus list in neural beta row order:

```bash
python experiments/build_bold5000_neural_order_stimuli_list.py \
  --output experiments/results/bold5000_stimuli_list_neural_order.txt
```

Run layerwise RSA:

```bash
python experiments/run_layerwise_hierarchy.py \
  --checkpoint <checkpoint.ckpt> \
  --config egomuscle/training/config.yaml \
  --neural-dir egomuscle/eval/neural_rdms \
  --stimuli-dir data/raw/bold5000/extracted/BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli \
  --stimuli-list experiments/results/bold5000_stimuli_list_neural_order.txt \
  --output experiments/results/layerwise_hierarchy.json
```
