#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

VIDEO_DIR="${VIDEO_DIR:-${SCRIPT_DIR}/../../data/images}"
TEMPLATE="${TEMPLATE:-${SCRIPT_DIR}/template_wilor.sh}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/generated_jobs}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/SLURM_logs}"
RECENT_LOG_COUNT="${RECENT_LOG_COUNT:-12}"
RECENT_TIME_MARGIN_HOURS="${RECENT_TIME_MARGIN_HOURS:-1.0}"
MAX_PARTITION_TIME="${MAX_PARTITION_TIME:-04:00:00}"

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

seconds_to_hms() {
    local total_seconds="$1"
    local hours minutes seconds
    hours=$(( total_seconds / 3600 ))
    minutes=$(( (total_seconds % 3600) / 60 ))
    seconds=$(( total_seconds % 60 ))
    printf '%02d:%02d:%02d\n' "${hours}" "${minutes}" "${seconds}"
}

time_to_seconds() {
    local value="$1"
    IFS=: read -r hours minutes seconds <<< "${value}"
    printf '%d\n' $((10#${hours} * 3600 + 10#${minutes} * 60 + 10#${seconds}))
}

recent_log_time_floor() {
    local logs=("${LOG_DIR}"/wilor-inference_*.out)
    [[ ${#logs[@]} -gt 0 ]] || return 0

    "${PYTHON_BIN}" - "${LOG_DIR}" "${RECENT_LOG_COUNT}" "${RECENT_TIME_MARGIN_HOURS}" <<'PY'
from pathlib import Path
import re
import sys

log_dir = Path(sys.argv[1])
recent_count = max(1, int(sys.argv[2]))
margin_hours = float(sys.argv[3])

logs = sorted(log_dir.glob("wilor-inference_*.out"))
if not logs:
    raise SystemExit(0)

pattern = re.compile(r"Execution took ([0-9]+(?:\.[0-9]+)?) hours")
hours = []
for path in logs[-recent_count:]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = pattern.findall(text)
    if matches:
        hours.append(float(matches[-1]))

if not hours:
    raise SystemExit(0)

estimated_seconds = int(round((max(hours) + margin_hours) * 3600))
print(estimated_seconds)
PY
}

mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

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
observed_floor_seconds=0

if OBSERVED_FLOOR_SECONDS=$(recent_log_time_floor); then
    if [[ -n "${OBSERVED_FLOOR_SECONDS}" ]]; then
        observed_floor_seconds="${OBSERVED_FLOOR_SECONDS}"
    fi
fi

max_partition_seconds=$(time_to_seconds "${MAX_PARTITION_TIME}")

if (( observed_floor_seconds > 0 )); then
    if (( observed_floor_seconds > max_partition_seconds )); then
        observed_floor_seconds="${max_partition_seconds}"
    fi
    echo "Using recent-log walltime floor: $(seconds_to_hms "${observed_floor_seconds}") from ${LOG_DIR}"
fi

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

    final_time="${TIME_FMT}"
    if (( observed_floor_seconds > 0 )); then
        estimated_seconds=$(time_to_seconds "${TIME_FMT}")
        if (( estimated_seconds < observed_floor_seconds )); then
            final_time=$(seconds_to_hms "${observed_floor_seconds}")
        fi
    fi

    job_script="${OUT_DIR}/demo_${name}.sh"
    escaped_name=$(escape_sed_replacement "${name}")
    escaped_time=$(escape_sed_replacement "${final_time}")

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
