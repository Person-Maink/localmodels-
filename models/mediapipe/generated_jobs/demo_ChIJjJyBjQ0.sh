#!/bin/bash
# ================ SLURM SETUP ================
# Available HPC Partitions:
#   compute / compute-p1   : CPU jobs (48 CPUs, 185 GB RAM, Phase 1)
#   compute-p2             : CPU jobs (64 CPUs, 250 GB RAM, Phase 2)
#   gpu / gpu-v100         : GPU jobs (4x V100, 32 GB VRAM each, Phase 1)
#   gpu-a100               : GPU jobs (4x A100, 80 GB VRAM each, Phase 2)
#   gpu-a100-small         : Small GPU jobs (<=1 GPU, <=10 GB VRAM, <=2 CPUs, <=4h)
#   memory                 : High-memory CPU jobs (>250 GB RAM)
#   visual                 : Visualization jobs

#SBATCH --job-name=mediapipe-inference
#SBATCH --partition=compute-p2
#SBATCH --time=02:31:46
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=%x.out
set -euo pipefail

# ================ OUTPUT FILES ================
base_name="${SLURM_JOB_NAME}"
dir="SLURM_logs"
count=$(printf "%03d" $(($(ls "$dir" 2>/dev/null | grep -c "^${base_name}_[0-9]\\+\\.out$") + 1)))
outfile="${dir}/${base_name}_${count}.out"
exec >"$outfile" 2>&1

module load 2024r1

PROJECT_ROOT="/scratch/mthakur/manifold"
MODEL_ROOT="${PROJECT_ROOT}/models/mediapipe"
COMMON_SH="${PROJECT_ROOT}/models/common/inference_common.sh"

source "${COMMON_SH}"

echo "Loaded modules:"
module list 2>&1

echo "================ SLURM JOB INFO ================"
echo "Job ID:         $SLURM_JOB_ID"
echo "Job Name:       $SLURM_JOB_NAME"
echo "Partition:      $SLURM_JOB_PARTITION"
echo "Node List:      $SLURM_JOB_NODELIST"
echo "CPUs per task:  $SLURM_CPUS_PER_TASK"
echo "Submit dir:     $SLURM_SUBMIT_DIR"
echo "Work dir:       $(pwd)"
echo "Video file:     ChIJjJyBjQ0.mp4"
echo "Job started at: $(date)"
start_time=$(date +%s)
echo "==============================================="

VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/data/images}"
VIDEO_NAME="${VIDEO_NAME:-ChIJjJyBjQ0}"
VIDEO_FILE="${VIDEO_FILE:-ChIJjJyBjQ0.mp4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/mediapipe}"
TARGET_FPS="${TARGET_FPS:-30}"
VISUALIZE="${VISUALIZE:-false}"
OVERWRITE="${OVERWRITE:-false}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-${MODEL_ROOT}/apptainer/template.sif}"

if ! VIDEO_PATH=$(resolve_video_path "${VIDEO_DIR}" "${VIDEO_NAME}" "${VIDEO_FILE}"); then
    echo "Could not resolve a supported video under ${VIDEO_DIR} for VIDEO_NAME='${VIDEO_NAME}' VIDEO_FILE='${VIDEO_FILE}'" >&2
    exit 1
fi

VIDEO_PATH=$(cd "$(dirname "${VIDEO_PATH}")" && pwd)/$(basename "${VIDEO_PATH}")
VIDEO_STEM="$(basename "${VIDEO_PATH%.*}")"
VIDEO_FILE_BASENAME="$(basename "${VIDEO_PATH}")"
MARKER_PATH="$(completion_marker_path "${OUTPUT_ROOT}" "${VIDEO_STEM}")"
CSV_PATH="${OUTPUT_ROOT}/keypoints/${VIDEO_STEM}_keypoints.csv"
VIDEO_OUTPUT_PATH="${OUTPUT_ROOT}/visualizations/${VIDEO_STEM}_overlay.mp4"

if [[ ! -f "${APPTAINER_IMAGE}" ]]; then
    echo "Apptainer image not found: ${APPTAINER_IMAGE}" >&2
    exit 1
fi

if [[ -f "${MARKER_PATH}" ]] && ! is_truthy "${OVERWRITE}"; then
    echo "Skipping ${VIDEO_STEM}: completion marker already exists at ${MARKER_PATH}"
    exit 0
fi

if is_truthy "${OVERWRITE}" || [[ ! -f "${MARKER_PATH}" ]]; then
    rm -f "${CSV_PATH}" "${VIDEO_OUTPUT_PATH}" "${MARKER_PATH}"
fi

visualize_flag="--no-visualize"
if is_truthy "${VISUALIZE}"; then
    visualize_flag="--visualize"
fi

container_cmd=$(cat <<EOF
set -euo pipefail
cd $(printf '%q' "${MODEL_ROOT}")
if [[ -x .venv/bin/python3 ]]; then
  PYTHON_BIN=.venv/bin/python3
else
  PYTHON_BIN=python
fi
"\${PYTHON_BIN}" main.py --video_folder $(printf '%q' "${VIDEO_DIR}") --video $(printf '%q' "${VIDEO_STEM}") --video_file $(printf '%q' "${VIDEO_FILE_BASENAME}") --output_folder $(printf '%q' "${OUTPUT_ROOT}") --target_fps $(printf '%q' "${TARGET_FPS}") ${visualize_flag}
EOF
)

echo "Mediapipe video: ${VIDEO_PATH}"
echo "Mediapipe output root: ${OUTPUT_ROOT}"
echo "Mediapipe completion marker: ${MARKER_PATH}"

srun apptainer exec \
  --bind /scratch:/scratch \
  "${APPTAINER_IMAGE}" \
  bash -lc "${container_cmd}"

if ! is_truthy "${VISUALIZE}"; then
    rm -f "${VIDEO_OUTPUT_PATH}"
fi
write_simple_completion_marker "${MARKER_PATH}" "mediapipe" "${VIDEO_STEM}" "${CSV_PATH}"

echo "==============================================="
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
hours=$(printf "%.2f" "$(echo "$elapsed/3600" | bc -l)")

echo "Job finished at: $(date)"
echo "Execution took $hours hours"
echo "Writing output to $outfile"
echo "Retained Mediapipe outputs: ${CSV_PATH}"
echo "==============================================="
