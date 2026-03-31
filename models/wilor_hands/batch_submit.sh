#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

VIDEO_DIR="${VIDEO_DIR:-${SCRIPT_DIR}/../../data/images}"
TEMPLATE="${TEMPLATE:-${SCRIPT_DIR}/template_wilor.sh}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/generated_jobs}"

escape_sed_replacement() {
    printf '%s\n' "$1" | sed 's/[\\&|]/\\&/g'
}

submit_job_script() {
    local job_script="$1"

    if [[ "${DRY_RUN:-0}" =~ ^([Tt][Rr][Uu][Ee]|[Yy][Ee]?[Ss]|[Oo][Nn]|1)$ ]]; then
        printf 'DRY_RUN: sbatch %q\n' "${job_script}"
    else
        sbatch "${job_script}"
    fi
}

mkdir -p "${OUT_DIR}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    :
elif [[ -x "${SCRIPT_DIR}/.venv/bin/python3.10" ]]; then
    PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python3.10"
elif [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

shopt -s nullglob nocaseglob
videos=("${VIDEO_DIR}"/*.mp4 "${VIDEO_DIR}"/*.avi "${VIDEO_DIR}"/*.mts "${VIDEO_DIR}"/*.mov)
video_count=0
submitted_count=0

for video in "${videos[@]}"; do
    [[ -f "${video}" ]] || continue

    filename=$(basename "${video}")
    name="${filename%.*}"

    if ! TIME_FMT=$("${PYTHON_BIN}" "${SCRIPT_DIR}/estimate_time.py" "${video}"); then
        echo "Skipping ${video}: failed to estimate runtime with ${PYTHON_BIN}"
        continue
    fi
    if [[ -z "${TIME_FMT}" ]]; then
        echo "Skipping ${video}: empty runtime estimate"
        continue
    fi

    job_script="${OUT_DIR}/demo_${name}.sh"
    escaped_name=$(escape_sed_replacement "${name}")
    escaped_time=$(escape_sed_replacement "${TIME_FMT}")

    sed \
        -e "s|__NAME__|${escaped_name}|g" \
        -e "s|__TIME__|${escaped_time}|g" \
        "${TEMPLATE}" > "${job_script}"

    video_count=$((video_count + 1))
    if ! submit_job_script "${job_script}"; then
        echo "Failed to submit ${job_script}; continuing with remaining jobs" >&2
        continue
    fi
    submitted_count=$((submitted_count + 1))
done

if (( video_count == 0 )); then
    echo "No videos found under ${VIDEO_DIR}"
    exit 0
fi

echo "Prepared ${video_count} wilor job scripts in ${OUT_DIR}"
echo "Submitted ${submitted_count} wilor jobs"
