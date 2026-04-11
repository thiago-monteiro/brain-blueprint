#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TWENTE_DL_DIR="${TWENTE_DL_DIR:-${ROOT}/data/raw/twente_zenodo_dl}"
EMG_EXTRACT="${TWENTE_EMG_EXTRACT:-${ROOT}/data/raw/twente_emg/extracted}"
RGB_EXTRACT="${TWENTE_RGB_EXTRACT:-${ROOT}/data/raw/twente_rgb/extracted}"

PROCESSED_URL="https://zenodo.org/api/records/6457662/files/Processed_data.rar/content"
RAW_URL="https://zenodo.org/api/records/6457662/files/Raw_data.rar/content"
VIDEOS_URL="https://zenodo.org/api/records/6644593/files/Videos_anonymized.rar/content"

mkdir -p "${TWENTE_DL_DIR}"

extract_rar_into() {
  local archive="$1"
  local dest="$2"
  mkdir -p "${dest}"
  if command -v unrar >/dev/null 2>&1; then
    ( cd "${dest}" && unrar x -o+ "${archive}" )
  elif command -v 7z >/dev/null 2>&1; then
    7z x -y "-o${dest}" "${archive}"
  else
    echo "Install unrar (recommended) or p7zip-full (7z) to extract .rar archives." >&2
    exit 1
  fi
}

download_file() {
  local url="$1"
  local out="$2"
  mkdir -p "$(dirname "${out}")"
  if [[ -f "${out}" ]]; then
    echo "Reusing existing ${out}"
    return
  fi
  echo "Downloading -> ${out}"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c \
      -x 16 -s 16 -k 1M \
      --file-allocation=none \
      --retry-wait=5 --max-tries=0 \
      --continue=true \
      -d "$(dirname "${out}")" \
      -o "$(basename "${out}")" \
      "${url}"
  else
    echo "aria2c not found, falling back to curl" >&2
    curl -fL --retry 3 --retry-delay 5 -C - -o "${out}.part" "${url}"
    mv -f "${out}.part" "${out}"
  fi
}

declare -A PIDS=()
declare -A DESTS=()

if [[ "${SKIP_PROCESSED:-0}" != "1" ]]; then
  download_file "${PROCESSED_URL}" "${TWENTE_DL_DIR}/Processed_data.rar" &
  PIDS[processed]=$!
  DESTS[processed]="${TWENTE_DL_DIR}/Processed_data.rar"
fi

if [[ "${SKIP_RAW:-1}" != "1" ]]; then
  download_file "${RAW_URL}" "${TWENTE_DL_DIR}/Raw_data.rar" &
  PIDS[raw]=$!
  DESTS[raw]="${TWENTE_DL_DIR}/Raw_data.rar"
fi

if [[ "${SKIP_VIDEOS:-0}" != "1" ]]; then
  download_file "${VIDEOS_URL}" "${TWENTE_DL_DIR}/Videos_anonymized.rar" &
  PIDS[videos]=$!
  DESTS[videos]="${TWENTE_DL_DIR}/Videos_anonymized.rar"
fi

FAILED=0
for key in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$key]}"; then
    echo "ERROR: download failed for ${key} (${DESTS[$key]})" >&2
    FAILED=1
  fi
done
[[ $FAILED -eq 1 ]] && exit 1

echo "All downloads complete."

if [[ "${SKIP_PROCESSED:-0}" != "1" ]]; then
  echo "Extracting Processed_data.rar -> ${EMG_EXTRACT} …"
  extract_rar_into "${TWENTE_DL_DIR}/Processed_data.rar" "${EMG_EXTRACT}"
  if [[ ! -d "${EMG_EXTRACT}/Processed_data" ]]; then
    echo "WARN: expected ${EMG_EXTRACT}/Processed_data after extract; check RAR layout." >&2
  fi
fi

if [[ "${SKIP_RAW:-1}" != "1" ]]; then
  echo "Extracting Raw_data.rar -> ${EMG_EXTRACT} …"
  extract_rar_into "${TWENTE_DL_DIR}/Raw_data.rar" "${EMG_EXTRACT}"
fi

if [[ "${SKIP_VIDEOS:-0}" != "1" ]]; then
  echo "Extracting Videos_anonymized.rar -> ${RGB_EXTRACT} …"
  extract_rar_into "${TWENTE_DL_DIR}/Videos_anonymized.rar" "${RGB_EXTRACT}"
  if [[ ! -d "${RGB_EXTRACT}/Videos_anonymized" ]]; then
    echo "WARN: expected ${RGB_EXTRACT}/Videos_anonymized after extract; check RAR layout." >&2
  fi
fi

echo ""
echo "Twente raw layout ready. Next (from repo root, venv active):"
echo "  python -m egomuscle.data.build_manifests \\"
echo "    --twente-emg-root ${EMG_EXTRACT} \\"
echo "    --twente-rgb-root ${RGB_EXTRACT} \\"
echo "    --output-dir data/processed/manifests"
echo "  python -m egomuscle.data.build_twente_dataset --view front"
echo "Eval default path: data/processed_real/twente (see build_twente_dataset --output-root)"