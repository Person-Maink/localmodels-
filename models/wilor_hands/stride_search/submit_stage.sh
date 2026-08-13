#!/usr/bin/env bash
# Submit every available HMP config for a stage; STAGE=stage1 or STAGE=stage2.
set -euo pipefail
SEARCH_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STAGE="${STAGE:-stage1}"
CONFIG_DIR="${SEARCH_ROOT}/configs/${STAGE}"
SUBMITTER_DIR="${SEARCH_ROOT}/submitters/${STAGE}"
[[ "${STAGE}" == "stage1" || "${STAGE}" == "stage2" ]] || { echo "STAGE must be stage1 or stage2" >&2; exit 2; }
shopt -s nullglob
configs=("${CONFIG_DIR}"/*.yaml)
if (( ${#configs[@]} == 0 )); then echo "No ${STAGE} configs found in ${CONFIG_DIR}. Add finalist YAMLs before submitting." >&2; exit 2; fi
mkdir -p "${SUBMITTER_DIR}"
for config_path in "${configs[@]}"; do
    config_id=$(basename "${config_path}" .yaml)
    submitter="${SUBMITTER_DIR}/submit_${config_id}.sh"
    if [[ ! -f "${submitter}" ]]; then
        cat > "${submitter}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
SEARCH_ROOT=\$(cd "\$(dirname "\${BASH_SOURCE[0]}")/../.." && pwd)
exec env STAGE='${STAGE}' CONFIG_ID='${config_id}' "\${SEARCH_ROOT}/submit_config.sh" "\$@"
EOF
        chmod +x "${submitter}"
    fi
    "${submitter}"
done
