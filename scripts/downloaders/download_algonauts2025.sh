#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_DIR="${ALGONAUTS_OUT_DIR:-data/raw/algonauts2025}"
DATALAD_URL="${ALGONAUTS_DATALAD_URL:-https://github.com/courtois-neuromod/algonauts_2025.competitors.git}"

usage() {
  cat <<EOF_USAGE
Download Algonauts 2025 fMRI (2.3 GB) + Movie10 stimuli (15-18 GB) via DataLad.

Requires DataLad (pip install datalad).
If DataLad is unavailable, set ALGONAUTS_CURL_FALLBACK=1 and
ALGONAUTS_CONP_URL to a CONP portal download URL.

Usage:
  bash scripts/downloaders/download_algonauts2025.sh
  ALGONAUTS_CURL_FALLBACK=1 ALGONAUTS_CONP_URL="https://..." bash scripts/downloaders/download_algonauts2025.sh

Environment:
  ALGONAUTS_OUT_DIR          Target directory (default: data/raw/algonauts2025)
  ALGONAUTS_DATALAD_URL      DataLad dataset URL
  ALGONAUTS_CURL_FALLBACK    If 1, use curl instead of datalad
  ALGONAUTS_CONP_URL         CONP portal download URL (for curl fallback)
EOF_USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

mkdir -p "$OUT_DIR"

if [[ "${ALGONAUTS_CURL_FALLBACK:-0}" != "1" ]]; then
  if ! command -v datalad &>/dev/null; then
    echo "DataLad not found. Install with: pip install datalad" >&2
    echo "Or set ALGONAUTS_CURL_FALLBACK=1 with ALGONAUTS_CONP_URL." >&2
    exit 1
  fi

  echo "=== DataLad install ==="
  datalad install -r -s "$DATALAD_URL" "$OUT_DIR"

  cd "$OUT_DIR"

  echo "=== Downloading fMRI (2.3 GB) ==="
  datalad get -r -J8 fmri/*

  echo "=== Downloading Movie10 stimuli (15-18 GB) ==="
  datalad get -r -J8 stimuli/movies/movie10/*

  echo "=== Done ==="
else
  CONP_URL="${ALGONAUTS_CONP_URL:-}"
  if [[ -z "$CONP_URL" ]]; then
    echo "ALGONAUTS_CURL_FALLBACK=1 requires ALGONAUTS_CONP_URL to be set." >&2
    exit 1
  fi
  echo "=== Curl fallback: downloading from CONP portal ==="
  curl -fL --retry 10 --retry-delay 5 --retry-connrefused -C - "$CONP_URL" -o "$OUT_DIR/algonauts2025.tar.gz"
  echo "Extracting..."
  tar -xzf "$OUT_DIR/algonauts2025.tar.gz" -C "$OUT_DIR"
  echo "=== Done ==="
fi

echo ""
echo "Download complete in: $OUT_DIR"
ls -lh "$OUT_DIR"
