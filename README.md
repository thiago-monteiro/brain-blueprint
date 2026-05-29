# EgoMuscle

Research code for **EgoMuscle**: joint modeling of **peripersonal or exocentric video** with **muscle activation** (simulated MinT / real EMG validation) to study whether embodied multimodal training yields representations that align with human **fMRI** via representational similarity analysis (RSA).

Use **`egomuscle/training/config.yaml`** for training defaults and paths.

## Repository layout

| Path | Purpose |
|------|--------|
| `egomuscle/` | Package: data pipelines, `EgoMuscle` model, training (PyTorch Lightning), eval (RSA, RDM, probes) |
| `experiments/` | Shell/Python drivers for ablations, RDM comparison, RSA benchmarks, dataset checks |
| `scripts/downloaders/download_videomae_weights.py` | Optional helper to materialize VideoMAE-Base weights under `models/videomae-base/` |
|`metadata.json` | Twente splits for reproducibility. |


Processed datasets, checkpoints, wandb runs, and most of `experiments/results/` are **gitignored**; see `.gitignore`.

## Setup

Python 3.11+ recommended. From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run modules with the repo root on `PYTHONPATH` (no `pyproject.toml` yet):

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Automated bootstrap

From the repo root:

```bash
bash scripts/setup_egomuscle.sh          # venv + pip + PYTHONPATH helper only
bash scripts/download_egomuscle_data.sh # VideoMAE weights, MinT (RADAR), Twente (Zenodo)
```

`setup_egomuscle.sh` does **not** download data. Use `download_egomuscle_data.sh` (or individual scripts under `scripts/`) for **VideoMAE**, **MinT**, and **Twente**; AMASS, BABEL, and SMPL+H still need manual steps. After those are available, `FULL_BUILD=1 bash scripts/setup_egomuscle.sh` runs manifests and the AMASS rendering pipeline into `data/processed/` (ego) and `data/processed_exo/` (exo).

Important:
- `data/processed/` is the canonical `E3` ego dataset in the current experiments. It is rendered by `egomuscle.data.egox_pipeline` with `camera_mode="peripersonal"`.
- `experiments/build_babel_viewpair.sh` does **not** produce the same view. Its `babel_recenter` output is a detector-based body-centered crop intended only as a BABEL recentered-vs-raw viewpair helper. Do not treat it as the `E3` peripersonal renderer.

Clean full run: `WANDB_INIT_TIMEOUT=180 ./experiments/run_clean_full_experiments.sh`.

## Training

Prepare processed data under `data/processed/...` (paths must match `egomuscle/training/config.yaml`). By default the config uses the Hugging Face model id `MCG-NJU/videomae-base`; optionally run the downloader to materialize weights under `models/videomae-base/`, then:

```bash
python -m egomuscle.training.train --config egomuscle/training/config.yaml
```

Overrides use dotted keys, for example:

```bash
python -m egomuscle.training.train --config egomuscle/training/config.yaml \
  --override training.max_epochs=10
```

## Experiments

Use `experiments/run_clean_full_experiments.sh` for the maintained end-to-end run, `experiments/run_scaling_law.sh` for scaling only, and `experiments/run_ablations_csv.py` for ablations only.
# Data and model prerequisites (not downloaded by `setup_egomuscle.sh`)

The automated script pulls **MinT** (open on RADAR) and **VideoMAE-Base** weights. Training the AMASS-rendered pipeline still needs the following.

## 1. AMASS

Register at [https://amass.is.tue.mpg.de](https://amass.is.tue.mpg.de), download the subsets you need, and unpack so that `.npz` sequences are discoverable under:

`data/raw/amass/`

Specifically, you need the following AMASS subsets for this pipeline:
- BMLmovi
- BMLrub
- EyesJapanDataset
- KIT
- TotalCapture

Paths must match what MinT / BABEL reference (same layout as the AMASS release).

## 2. BABEL annotations

Download the JSON splits from [https://babel.is.tue.mpg.de/data.html](https://babel.is.tue.mpg.de/data.html) (academic licence). Place:

`data/raw/babel/babel_v1.0_release/train.json`  
`data/raw/babel/babel_v1.0_release/val.json`  
`data/raw/babel/babel_v1.0_release/test.json`  
(and, when available, `extra_train.json`, `extra_val.json`)

`python -m egomuscle.data.build_manifests` joins BABEL URLs to MinT sequences; rows without BABEL are dropped when building the video dataset.

## 3. SMPL+H body models

Download the SMPL+H models from the official MANO site. Specifically, download the "**SMPLH model version ready to load by the smplx python package**" (usually downloads as `smplx.zip`). Extract the male and female files, and create a symlink for the neutral model (as a fallback for ungendered AMASS clips):

`data/raw/models/smplh/SMPLH_FEMALE.pkl`  
`data/raw/models/smplh/SMPLH_MALE.pkl`  
`data/raw/models/smplh/SMPLH_NEUTRAL.pkl` (You can create this as a symlink to `SMPLH_MALE.pkl`)

Then either export `SMPL_MODEL_DIR` pointing at `data/raw/models`, or pass `--smpl-model-root data/raw/models` to `build_min_t_dataset` (see `experiments/build_amass_pair.py`).

## 4. Build processed clips

With MinT + BABEL + AMASS + SMPL in place:

```bash
source .venv/bin/activate && source scripts/activate_egomuscle.sh
FULL_BUILD=1 bash scripts/setup_egomuscle.sh
```

Or manually:

```bash
python -m egomuscle.data.build_manifests \
  --mint-root data/raw/mint_extracted \
  --babel-root data/raw/babel \
  --output-dir data/processed/manifests

AMASS_ROOT=data/raw/amass SMPL_ROOT=data/raw/models \
  python experiments/build_amass_pair.py data/processed data/processed_exo
```

Smoke test (few clips, faster):

```bash
MAX_RECORDS=20 FULL_BUILD=1 bash scripts/setup_egomuscle.sh
```

## 5. Twente real EMG

Automated download + extract (needs `curl` and `unrar` or `7z`):

```bash
bash scripts/download_twente.sh
```

Or use `bash scripts/download_egomuscle_data.sh` to fetch Twente together with VideoMAE and MinT. Then build the eval layout:

```bash
python -m egomuscle.data.build_manifests \
  --twente-emg-root data/raw/twente_emg/extracted \
  --twente-rgb-root data/raw/twente_rgb/extracted \
  --output-dir data/processed/manifests
python -m egomuscle.data.build_twente_dataset --view front
```

Eval default: `data/processed_real/twente` (see `egomuscle/eval/twente_eval.py`). Zenodo records: [6457662](https://zenodo.org/records/6457662) (EMG), [6644593](https://zenodo.org/records/6644593) (RGB).

## 6. BOLD5000 neural RSA

BOLD5000 is used for the layerwise neural-RSA followups and SMFE posthoc BOLD5000 tables. The official download page is:

[https://bold5000-dataset.github.io/website/download.html](https://bold5000-dataset.github.io/website/download.html)

The BOLD5000 page lists:
- a stimuli/images download with labels;
- Release 2.0 as the recommended processed GLMsingle dataset on Kilthub/Figshare;
- Release 1.0 resources including ROI brain responses, raw/scanner data, physiological/behavioral data, scanning protocols, 5,254 image names, labels, and presentation code;
- OpenNeuro access for BIDS/fMRIPREP-style MRI data, ROI masks, and the 5,254 image names.

For this repo's RSA pipeline, you need two pieces:

1. **GLMsingle ROI beta matrices** for neural RDM construction.
2. **Stimulus images plus presentation lists** so model activations are computed in the same 5,254-trial order as the neural betas.

Expected local layout:

`data/raw/bold5000/extracted/BOLD5000_GLMsingle_ROI_betas/mat/`  
`data/raw/bold5000/extracted/BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli/`  
`data/raw/bold5000/extracted/BOLD5000_Stimuli/Stimuli_Presentation_Lists/`

Download the ROI betas with the helper script:

```bash
bash scripts/download_bold5000_roi_betas.sh
unzip data/raw/bold5000/BOLD5000_GLMsingle_ROI_betas.zip -d data/raw/bold5000/extracted
```

Download the BOLD5000 stimuli/images archive from the official BOLD5000 page and extract it so `BOLD5000_Stimuli` is under `data/raw/bold5000/extracted/`.

Build neural RDMs from the ROI betas:

```bash
python -m egomuscle.eval.prepare_bold5000_rdms \
  --mat-root data/raw/bold5000/extracted/BOLD5000_GLMsingle_ROI_betas/mat \
  --output-dir egomuscle/eval/neural_rdms
```

Build the stimulus list in neural beta row order. Do not glob the image directory for this step: BOLD5000 has repeated stimuli, and the RSA matrix expects the 5,254-trial presentation order.

```bash
python experiments/build_bold5000_neural_order_stimuli_list.py \
  --output experiments/results/bold5000_stimuli_list_neural_order.txt
```

Layerwise RSA can then use:

```bash
python experiments/run_layerwise_hierarchy.py \
  --checkpoint <checkpoint.ckpt> \
  --config egomuscle/training/config.yaml \
  --neural-dir egomuscle/eval/neural_rdms \
  --stimuli-dir data/raw/bold5000/extracted/BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli \
  --stimuli-list experiments/results/bold5000_stimuli_list_neural_order.txt \
  --output experiments/results/layerwise_hierarchy.json
```
