#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

URL="${BOLD5000_URL:-https://figshare.com/ndownloader/articles/14456124/versions/2}"
API_URL="${BOLD5000_API_URL:-https://api.figshare.com/v2/articles/14456124/versions/2}"
OUT_DIR="${BOLD5000_OUT_DIR:-data/raw/bold5000}"
OUT_FILE="${BOLD5000_OUT_FILE:-BOLD5000_GLMsingle_ROI_betas.zip}"

usage() {
  cat <<EOF
Download BOLD5000 GLMsingle ROI betas from Figshare.

Usage:
  bash scripts/download_bold5000_roi_betas.sh
  bash scripts/download_bold5000_roi_betas.sh --out-dir /path/to/downloads
  bash scripts/download_bold5000_roi_betas.sh --output roi_betas.zip
  bash scripts/download_bold5000_roi_betas.sh --url https://figshare.com/ndownloader/articles/14456124/versions/2

Environment overrides:
  BOLD5000_URL
  BOLD5000_API_URL
  BOLD5000_OUT_DIR
  BOLD5000_OUT_FILE
EOF
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
    --api-url)
      API_URL="$2"
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

resolve_download_url() {
  python - "$API_URL" <<'PY'
import json
import sys
import urllib.request

api_url = sys.argv[1]
req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as resp:
    payload = json.load(resp)

files = payload.get("files", [])
if not files:
    raise SystemExit("No files listed in Figshare article metadata.")

target = None
for f in files:
    name = (f.get("name") or "").lower()
    if "glm" in name and "roi" in name and name.endswith(".zip"):
        target = f
        break
if target is None:
    target = files[0]

download_url = target.get("download_url")
if not download_url:
    raise SystemExit("Selected Figshare file has no download_url.")

print(download_url)
PY
}

echo "Downloading BOLD5000 ROI betas"
echo "URL:      $URL"
echo "API URL:  $API_URL"
echo "Dest: $DEST"
echo

DOWNLOAD_URL="$(resolve_download_url)"
echo "Resolved file URL from Figshare API."
echo

if command -v curl >/dev/null 2>&1; then
  curl -fL --retry 10 --retry-delay 5 --retry-connrefused -C - "$DOWNLOAD_URL" -o "$DEST"
elif command -v wget >/dev/null 2>&1; then
  wget -c --tries=10 --waitretry=5 "$DOWNLOAD_URL" -O "$DEST"
else
  echo "Neither curl nor wget is available. Please install one of them." >&2
  exit 1
fi

echo
echo "Download complete:"
echo "  $DEST"
echo "Next step:"
echo "  unzip \"$DEST\" -d data/raw/bold5000/extracted"
