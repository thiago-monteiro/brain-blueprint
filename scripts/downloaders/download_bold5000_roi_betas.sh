#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

URL="${BOLD5000_URL:-https://figshare.com/ndownloader/articles/14456124/versions/2}"
OUT_DIR="${BOLD5000_OUT_DIR:-data/raw/bold5000}"
OUT_FILE="${BOLD5000_OUT_FILE:-BOLD5000_GLMsingle_ROI_betas.zip}"

usage() {
  cat <<EOF_USAGE
Download BOLD5000 GLMsingle ROI betas from Figshare.

Usage:
  bash scripts/downloaders/download_bold5000_roi_betas.sh
  bash scripts/downloaders/download_bold5000_roi_betas.sh --out-dir /path/to/downloads
  bash scripts/downloaders/download_bold5000_roi_betas.sh --output roi_betas.zip
  bash scripts/downloaders/download_bold5000_roi_betas.sh --url https://figshare.com/ndownloader/articles/14456124/versions/2

Environment overrides:
  BOLD5000_URL
  BOLD5000_OUT_DIR
  BOLD5000_OUT_FILE
EOF_USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url)
      URL="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --output)
      OUT_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

mkdir -p "$OUT_DIR"
DEST="${OUT_DIR%/}/$OUT_FILE"

echo "Downloading BOLD5000 ROI betas"
echo "URL:  $URL"
echo "Dest: $DEST"
echo

curl -fL --retry 10 --retry-delay 5 --retry-connrefused -C - "$URL" -o "$DEST"

echo
echo "Download complete:"
echo "  $DEST"
echo "Next step:"
echo "  unzip \"$DEST\" -d data/raw/bold5000/extracted"
