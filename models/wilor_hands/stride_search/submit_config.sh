#!/usr/bin/env bash
# Submit all selected videos for one STRIDE sweep configuration.
set -euo pipefail
SEARCH_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_ROOT=$(cd "${SEARCH_ROOT}/.." && pwd)
PROJECT_ROOT=$(cd "${MODEL_ROOT}/../.." && pwd)
STAGE="${STAGE:-stage1}"
CONFIG_ID="${CONFIG_ID:?Set CONFIG_ID to a YAML stem in configs/${STAGE}.}"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/data/test}"
WILOR_CACHE_ROOT="${WILOR_CACHE_ROOT:-${PROJECT_ROOT}/outputs/wilor}"
SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/stride_search}"
FRAME_CACHE_ROOT="${FRAME_CACHE_ROOT:-${VIDEO_DIR}}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-${MODEL_ROOT}/apptainer/template-hmp.sif}"
SPLIT_PATH="${SEARCH_ROOT}/splits/${STAGE}.json"
CONFIG_PATH="${SEARCH_ROOT}/configs/${STAGE}/${CONFIG_ID}.yaml"
JOB_ROOT="${SEARCH_ROOT}/generated_jobs/${STAGE}/${CONFIG_ID}"
LOG_ROOT="${MODEL_ROOT}/SLURM_logs/stride_search/${STAGE}/${CONFIG_ID}"
OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT}/${STAGE}/${CONFIG_ID}"
[[ "${STAGE}" == "stage1" || "${STAGE}" == "stage2" ]] || { echo "STAGE must be stage1 or stage2" >&2; exit 2; }
[[ -f "${SPLIT_PATH}" && -f "${CONFIG_PATH}" ]] || { echo "Missing split or config for ${STAGE}/${CONFIG_ID}" >&2; exit 2; }
mkdir -p "${JOB_ROOT}" "${LOG_ROOT}" "${OUTPUT_ROOT}"
if ! VIDEO_TEXT=$(python3 - "${SPLIT_PATH}" "${VIDEO_DIR}" "${WILOR_CACHE_ROOT}" <<'PY'
import json, sys
from pathlib import Path
split_path, video_root, wilor_root = map(Path, sys.argv[1:])
videos = json.loads(split_path.read_text(encoding="utf-8")).get("video_ids")
if not isinstance(videos, list) or not videos or len(videos) != len(set(videos)):
    raise SystemExit(f"Invalid or duplicate video_ids in {split_path}")
for video in videos:
    if not isinstance(video, str) or not (video_root / f"{video}.mp4").is_file(): raise SystemExit(f"Missing video: {video}")
    if not (wilor_root / video / "meshes").is_dir(): raise SystemExit(f"Missing cached WiLoR meshes: {video}")
print("\n".join(videos))
PY
); then
    echo "Aborting ${STAGE}/${CONFIG_ID}: split input validation failed." >&2
    exit 1
fi
mapfile -t VIDEOS <<< "${VIDEO_TEXT}"
planned=0; skipped=0
for video in "${VIDEOS[@]}"; do
    marker="${OUTPUT_ROOT}/_completed/${video}.json"
    if [[ -f "${marker}" && "${OVERWRITE:-false}" != "true" ]]; then echo "Skipping completed ${CONFIG_ID}/${video}"; skipped=$((skipped + 1)); continue; fi
    job_path="${JOB_ROOT}/${video}.sh"
    cat > "${job_path}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=stride-${STAGE}-${CONFIG_ID}
#SBATCH --partition=gpu-a100-small
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=10G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=${LOG_ROOT}/%x_%j.out
set -euo pipefail
export VIDEO_DIR='${VIDEO_DIR}' VIDEO_NAME='${video}' WILOR_CACHE_ROOT='${WILOR_CACHE_ROOT}' OUTPUT_ROOT='${OUTPUT_ROOT}' FRAME_CACHE_ROOT='${FRAME_CACHE_ROOT}' STRIDE_BACKEND='hmp' STRIDE_CONFIG_PATH='${CONFIG_PATH}' APPTAINER_IMAGE='${APPTAINER_IMAGE}' OVERWRITE='${OVERWRITE:-false}'
exec bash '${MODEL_ROOT}/stride_inference.sh'
EOF
    chmod +x "${job_path}"
    if [[ "${DRY_RUN:-false}" == "true" ]]; then echo "DRY_RUN: sbatch ${job_path}"; else sbatch "${job_path}"; fi
    planned=$((planned + 1))
done
echo "${STAGE}/${CONFIG_ID}: planned=${planned}, skipped=${skipped}, selected=${#VIDEOS[@]}"
