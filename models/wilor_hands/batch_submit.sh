#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../slurm_array_common.sh"

VIDEO_DIR="${VIDEO_DIR:-${SCRIPT_DIR}/../../data/images}"
TEMPLATE="${TEMPLATE:-${SCRIPT_DIR}/template_wilor.sh}"
ARRAY_TEMPLATE="${ARRAY_TEMPLATE:-${SCRIPT_DIR}/template_wilor_array.sh}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/generated_jobs}"
DEBUG_DIR="${DEBUG_DIR:-${OUT_DIR}/debug}"
ARRAY_DIR="${ARRAY_DIR:-${OUT_DIR}/arrays}"
MANIFEST_DIR="${MANIFEST_DIR:-${OUT_DIR}/manifests}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
ARRAY_MAX_PARALLEL=$(normalize_positive_int "${ARRAY_MAX_PARALLEL:-8}" 8)
BUCKET_SECONDS=$(normalize_positive_int "${BUCKET_SECONDS:-1800}" 1800)

mkdir -p "${DEBUG_DIR}" "${ARRAY_DIR}" "${MANIFEST_DIR}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    :
elif [[ -x "${SCRIPT_DIR}/.venv/bin/python3.10" ]]; then
    PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python3.10"
elif [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

declare -A BUCKET_MANIFESTS=()
declare -A BUCKET_TIMES=()
video_count=0
submitted_arrays=0

shopt -s nullglob
for video in "${VIDEO_DIR}"/*.*; do
    [[ -f "${video}" ]] || continue

    filename=$(basename "${video}")
    name="${filename%.*}"
    ext="${filename##*.}"
    safe_name=$(safe_job_name "${filename}")

    if ! TIME_FMT=$("${PYTHON_BIN}" "${SCRIPT_DIR}/estimate_time.py" "${video}"); then
        echo "Skipping ${video}: failed to estimate runtime with ${PYTHON_BIN}"
        continue
    fi
    if [[ -z "${TIME_FMT}" ]]; then
        echo "Skipping ${video}: empty runtime estimate"
        continue
    fi

    debug_script="${DEBUG_DIR}/demo_${safe_name}.sh"
    escaped_name=$(escape_sed_replacement "${name}")
    escaped_time=$(escape_sed_replacement "${TIME_FMT}")

    sed \
        -e "s|__NAME__|${escaped_name}|g" \
        -e "s|__TIME__|${escaped_time}|g" \
        "${TEMPLATE}" > "${debug_script}"
    chmod +x "${debug_script}"

    time_seconds=$(hms_to_seconds "${TIME_FMT}")
    bucket_total_seconds=$(round_up_bucket_seconds "${time_seconds}" "${BUCKET_SECONDS}")
    bucket_time=$(seconds_to_hms "${bucket_total_seconds}")
    bucket_label="${bucket_time//:/-}"

    if [[ -z "${BUCKET_MANIFESTS[${bucket_label}]+x}" ]]; then
        BUCKET_MANIFESTS["${bucket_label}"]="${MANIFEST_DIR}/bucket_${bucket_label}_${RUN_STAMP}.tsv"
        BUCKET_TIMES["${bucket_label}"]="${bucket_time}"
        : > "${BUCKET_MANIFESTS[${bucket_label}]}"
    fi

    printf '%s\t%s\t%s\t%s\n' "${safe_name}" "${name}" "${ext}" "${filename}" >> "${BUCKET_MANIFESTS[${bucket_label}]}"
    video_count=$((video_count + 1))
done

if (( video_count == 0 )); then
    echo "No videos found under ${VIDEO_DIR}"
    exit 0
fi

while IFS= read -r bucket_label; do
    [[ -n "${bucket_label}" ]] || continue

    manifest="${BUCKET_MANIFESTS[${bucket_label}]}"
    bucket_time="${BUCKET_TIMES[${bucket_label}]}"
    task_count=$(wc -l < "${manifest}")
    task_count="${task_count//[[:space:]]/}"
    last_index=$((task_count - 1))
    array_script="${ARRAY_DIR}/submit_${bucket_label}_${RUN_STAMP}.sh"
    array_spec="0-${last_index}%${ARRAY_MAX_PARALLEL}"

    sed \
        -e "s|__TIME__|$(escape_sed_replacement "${bucket_time}")|g" \
        -e "s|__ARRAY_SPEC__|$(escape_sed_replacement "${array_spec}")|g" \
        -e "s|__MANIFEST_FILE__|$(escape_sed_replacement "$(basename "${manifest}")")|g" \
        "${ARRAY_TEMPLATE}" > "${array_script}"
    chmod +x "${array_script}"

    echo "Submitting wilor bucket ${bucket_time} with ${task_count} videos"
    submit_batch_script "${array_script}"
    submitted_arrays=$((submitted_arrays + 1))
done < <(printf '%s\n' "${!BUCKET_MANIFESTS[@]}" | sort)

echo "Prepared ${video_count} wilor videos across ${submitted_arrays} array submissions"
echo "Debug scripts: ${DEBUG_DIR}"
echo "Array scripts: ${ARRAY_DIR}"
echo "Manifests: ${MANIFEST_DIR}"
