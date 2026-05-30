#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DO_VIDEOMAE=1
DO_MINT=1
DO_TWENTE=1

for arg in "$@"; do
  case "${arg}" in
    --videomae-only) DO_MINT=0; DO_TWENTE=0 ;;
    --mint-only) DO_VIDEOMAE=0; DO_TWENTE=0 ;;
    --twente-only) DO_VIDEOMAE=0; DO_MINT=0 ;;
    --skip-videomae) DO_VIDEOMAE=0 ;;
    --skip-mint) DO_MINT=0 ;;
    --skip-twente) DO_TWENTE=0 ;;
    -h|--help)
      cat <<'EOF'
Usage: bash scripts/downloaders/download_egomuscle_data.sh [options]

Options:
  --videomae-only   Download VideoMAE weights only.
  --mint-only       Download MinT only.
  --twente-only     Download Twente EMG/RGB only.
  --skip-videomae   Skip VideoMAE weights.
  --skip-mint       Skip MinT.
  --skip-twente     Skip Twente.
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: ${arg}" >&2
      exit 1
      ;;
  esac
done

if [[ "${DO_VIDEOMAE}" == "1" ]]; then
  python "${ROOT}/scripts/downloaders/download_videomae_weights.py"
fi
if [[ "${DO_MINT}" == "1" ]]; then
  python "${ROOT}/scripts/downloaders/fetch_mint_dataset.py"
fi
if [[ "${DO_TWENTE}" == "1" ]]; then
  bash "${ROOT}/scripts/downloaders/download_twente.sh"
fi

echo "download_egomuscle_data.sh finished."
