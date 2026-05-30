#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ACTIVATE_HELPER="${ROOT}/scripts/activate_egomuscle.sh"
cat >"${ACTIVATE_HELPER}" <<'EOS'
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${PWD}"
EOS
chmod +x "${ACTIVATE_HELPER}" 2>/dev/null || true

if [[ "${SKIP_VENV:-0}" != "1" ]]; then
  if [[ ! -d .venv ]]; then
    python3 -m venv .venv
  fi
  source .venv/bin/activate
  pip install -U pip wheel
  pip install -r requirements.txt
else
  if [[ -d .venv ]]; then
    source .venv/bin/activate
  fi
fi

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${ROOT}"

echo ""
echo "=== Local setup finished (venv + PYTHONPATH helper) ==="
echo "  source .venv/bin/activate && source scripts/activate_egomuscle.sh"
echo "  Download public assets: bash scripts/downloaders/download_egomuscle_data.sh"
echo "  Manual/licensed data: see README.md"
echo ""

have_amass() {
  [[ -d "${ROOT}/data/raw/amass" ]] || return 1
  [[ -n "$(find "${ROOT}/data/raw/amass" -name "*.npz" 2>/dev/null | head -n1)" ]]
}

have_babel() {
  [[ -f "${ROOT}/data/raw/babel/babel_v1.0_release/train.json" ]]
}

have_smpl() {
  local d="${SMPL_MODEL_DIR:-${ROOT}/data/raw/models}"
  [[ -f "${d}/smplh/SMPLH_FEMALE.pkl" ]] || [[ -f "${d}/smplh/SMPLH_MALE.pkl" ]]
}

if [[ "${FULL_BUILD:-0}" == "1" ]]; then
  if ! have_babel; then
    echo "FULL_BUILD=1 but BABEL json not found at data/raw/babel/babel_v1.0_release/train.json" >&2
    exit 1
  fi
  if ! have_amass; then
    echo "FULL_BUILD=1 but no AMASS .npz under data/raw/amass" >&2
    exit 1
  fi
  if ! have_smpl; then
    echo "FULL_BUILD=1 but SMPL+H not found. Set SMPL_MODEL_DIR or install under data/raw/models/smplh/" >&2
    exit 1
  fi
  python -m egomuscle.data.build_manifests \
    --mint-root "${ROOT}/data/raw/mint_extracted" \
    --babel-root "${ROOT}/data/raw/babel" \
    --output-dir "${ROOT}/data/processed/manifests"

  BUILD_EXTRA=()
  if [[ -n "${MAX_RECORDS:-}" ]]; then
    BUILD_EXTRA+=(--max-records "${MAX_RECORDS}")
  fi
  AMASS_ROOT="${AMASS_ROOT:-${ROOT}/data/raw/amass}"
  SMPL_ROOT="${SMPL_ROOT:-${ROOT}/data/raw/models}"
  export AMASS_ROOT SMPL_ROOT
  python "${ROOT}/experiments/build_amass_pair.py" "${ROOT}/data/processed" "${ROOT}/data/processed_exo" "${BUILD_EXTRA[@]}"
  echo "Processed ego dataset: ${ROOT}/data/processed (matches egomuscle/training/config.yaml)"
else
  if have_babel && have_amass && have_smpl; then
    echo "AMASS, BABEL, and SMPL+H detected — you can run: FULL_BUILD=1 bash scripts/setup_egomuscle.sh"
  fi
fi
