#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
      grep '^#' "$0" | head -n 20
      exit 0
      ;;
    *)
      echo "Unknown option: ${arg}" >&2
      exit 1
      ;;
  esac
done

if [[ "${DO_VIDEOMAE}" == "1" ]]; then
  python "${ROOT}/downloaders/download_videomae_weights.py"
fi
if [[ "${DO_MINT}" == "1" ]]; then
  python "${ROOT}/downloaders/fetch_mint_dataset.py"
fi
if [[ "${DO_TWENTE}" == "1" ]]; then
  bash "${ROOT}/downloaders/download_twente.sh"
fi

echo "download_egomuscle_data.sh finished."
