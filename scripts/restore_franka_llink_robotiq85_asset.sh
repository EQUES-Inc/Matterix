#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_REPO="${PROJECT_ROOT}/source/matterix_assets/data"
ASSET_REV="618828b2e521fc3cf21bf8efb8bab580b44a7f69"
ASSET_DIR="robots/franka/franka-llink-robotiq85"

if [[ ! -d "${ASSET_REPO}/.git" && ! -f "${ASSET_REPO}/.git" ]]; then
    echo "[ERROR] Matterix asset submodule is not initialized: ${ASSET_REPO}" >&2
    exit 1
fi

echo "[INFO] Restoring the L-link Robotiq asset from ${ASSET_REV}"
git -C "${ASSET_REPO}" restore \
    --source="${ASSET_REV}" \
    -- "${ASSET_DIR}/franka-llink-robotiq85-inst.usda" \
       "${ASSET_DIR}/files/Materials" \
       "${ASSET_DIR}/files/Props" \
       "${ASSET_DIR}/files/meshes" \
       "${ASSET_DIR}/files/panda_robotiq_inst_col.usda" \
       "${ASSET_DIR}/files/panda_robotiq_inst_vis.usda"

ROOT_USDA="${ASSET_REPO}/${ASSET_DIR}/franka-llink-robotiq85-inst.usda"
sed -i \
    's#</robotiq_coupling_link_visuals>#</robotiq_85_coupling_link_visuals>#' \
    "${ROOT_USDA}"

echo "[OK] Restored ${ROOT_USDA}"