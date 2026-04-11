#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-data/raw}"
mkdir -p "${ROOT_DIR}"

cat <<EOF
EgoMuscle download scaffold

This project depends on external datasets with their own hosting and access rules:

- MinT:   https://simplexsigil.github.io/mint
- Twente EMG: https://zenodo.org/records/6457662
- Twente RGB: https://zenodo.org/records/6644593
- BOLD5000 Release 2.0 ROI betas: https://figshare.com/articles/dataset/BOLD5000_Release_2_0/14456124
- AMASS:  https://amass.is.tue.mpg.de

Suggested extraction layout:

${ROOT_DIR}/mint
${ROOT_DIR}/twente_emg
${ROOT_DIR}/twente_rgb
${ROOT_DIR}/bold5000
${ROOT_DIR}/amass

The Twente files currently published on Zenodo are:
- Processed_data.rar and Raw_data.rar under the EMG/multimodal record
- Videos_anonymized.rar under the RGB record
- BOLD5000_GLMsingle_ROI_betas.zip under the Release 2.0 figshare record

Then point training/config.yaml at your processed train/val/test splits.
EOF
