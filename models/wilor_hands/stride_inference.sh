#!/bin/bash
# ================ SLURM SETUP ================
# Available HPC Partitions:
#   compute / compute-p1   : CPU jobs (48 CPUs, 185 GB RAM, Phase 1)
#   compute-p2             : CPU jobs (64 CPUs, 250 GB RAM, Phase 2)
#   gpu / gpu-v100         : GPU jobs (4x V100, 32 GB VRAM each, Phase 1)
#   gpu-a100               : GPU jobs (4x A100, 80 GB VRAM each, Phase 2)
#   gpu-a100-small         : Small GPU jobs (≤1 GPU, ≤10 GB VRAM, ≤2 CPUs, ≤4h)
#   memory                 : High-memory CPU jobs (>250 GB RAM)
#   visual                 : Visualization jobs

set -euo pipefail

# ================ OUTPUT FILES ================
base_name="${SLURM_JOB_NAME:-stride-inference}"
dir="SLURM_logs"
mkdir -p "$dir"
count=$(printf "%03d" $(($(ls "$dir" 2>/dev/null | grep -c "^${base_name}_[0-9]\+\.out$") + 1)))
outfile="${dir}/${base_name}_${count}.out"
exec >"$outfile" 2>&1

# ================ SLURM SETUP ================
module load 2024r1
module load cuda/11.7

PROJECT_ROOT="/scratch/mthakur/manifold"
MODEL_ROOT="${PROJECT_ROOT}/models/wilor_hands"
COMMON_PY="${PROJECT_ROOT}/models/common/extract_video_frames.py"
COMMON_SH="${PROJECT_ROOT}/models/common/inference_common.sh"

source "${COMMON_SH}"

echo "Loaded modules:"
module list 2>&1

nvidia-smi

echo "================ SLURM JOB INFO ================"
echo "Job ID:         ${SLURM_JOB_ID:-n/a}"
echo "Job Name:       ${SLURM_JOB_NAME:-stride-inference}"
echo "Partition:      ${SLURM_JOB_PARTITION:-n/a}"
echo "Node List:      ${SLURM_JOB_NODELIST:-n/a}"
echo "CPUs per task:  ${SLURM_CPUS_PER_TASK:-n/a}"
echo "GPUs per task:  ${SLURM_GPUS_PER_TASK:-n/a}"
echo "Memory per GPU: ${SLURM_MEM_PER_GPU:-n/a}"
echo "Submit dir:     ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "Work dir:       $(pwd)"
echo "Job started at: $(date)"
start_time=$(date +%s)
echo "==============================================="

VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/data/test/me}"
VIDEO_NAME="${VIDEO_NAME:-me 1}"
VIDEO_FILE="${VIDEO_FILE:-me 1.mp4}"
WILOR_CACHE_ROOT="${WILOR_CACHE_ROOT:-${PROJECT_ROOT}/outputs/wilor}"
STRIDE_OUTPUT_ROOT="${STRIDE_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/stride}"
VIPE_OUTPUT_ROOT="${VIPE_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/vipe}"
HMP_ASSETS_ROOT="${HMP_ASSETS_ROOT:-${MODEL_ROOT}/_DATA/hmp_model}"
MANO_MODEL_PATH="${MANO_MODEL_PATH:-${MODEL_ROOT}/mano_data}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-${MODEL_ROOT}/apptainer/template.sif}"
OVERWRITE="${OVERWRITE:-false}"
KEEP_TEMP_FRAMES="${KEEP_TEMP_FRAMES:-false}"
VISUALIZE="${VISUALIZE:-false}"
USE_GPU="${USE_GPU:-true}"
TARGET_HAND="${TARGET_HAND:-auto}"
TEMP_PARENT="${TEMP_ROOT:-$(pick_temp_root)}"

if ! VIDEO_PATH=$(resolve_video_path "${VIDEO_DIR}" "${VIDEO_NAME}" "${VIDEO_FILE}"); then
    echo "Could not resolve a supported video under ${VIDEO_DIR} for VIDEO_NAME='${VIDEO_NAME}' VIDEO_FILE='${VIDEO_FILE}'" >&2
    exit 1
fi

VIDEO_PATH=$(cd "$(dirname "${VIDEO_PATH}")" && pwd)/$(basename "${VIDEO_PATH}")
VIDEO_STEM="$(basename "${VIDEO_PATH%.*}")"
MARKER_PATH="$(completion_marker_path "${STRIDE_OUTPUT_ROOT}" "${VIDEO_STEM}")"
VIDEO_OUTPUT_DIR="${STRIDE_OUTPUT_ROOT}/${VIDEO_STEM}"
VIDEO_OUTPUT_FILE="${STRIDE_OUTPUT_ROOT}/videos/${VIDEO_STEM}.mp4"

if [[ ! -f "${APPTAINER_IMAGE}" ]]; then
    echo "Apptainer image not found: ${APPTAINER_IMAGE}" >&2
    exit 1
fi

if [[ -f "${MARKER_PATH}" ]] && ! is_truthy "${OVERWRITE}"; then
    echo "Skipping ${VIDEO_STEM}: completion marker already exists at ${MARKER_PATH}"
    exit 0
fi

if is_truthy "${OVERWRITE}" || [[ ! -f "${MARKER_PATH}" ]]; then
    rm -rf "${VIDEO_OUTPUT_DIR}"
    rm -f "${VIDEO_OUTPUT_FILE}"
    rm -f "${MARKER_PATH}"
fi

mkdir -p "${STRIDE_OUTPUT_ROOT}"
mkdir -p "${TEMP_PARENT}"
TEMP_WORKDIR=$(mktemp -d "${TEMP_PARENT%/}/stride_${VIDEO_STEM}.XXXXXX")
TEMP_FRAME_ROOT="${TEMP_WORKDIR}/frames"
TEMP_FRAME_DIR="${TEMP_FRAME_ROOT}/${VIDEO_STEM}_frames"

cleanup_temp() {
    if is_truthy "${KEEP_TEMP_FRAMES}"; then
        echo "Keeping temporary STRIDE frames at ${TEMP_WORKDIR}"
        return
    fi
    rm -rf "${TEMP_WORKDIR}"
}
trap cleanup_temp EXIT

visualize_flag="--no-visualize"
if is_truthy "${VISUALIZE}"; then
    visualize_flag="--visualize"
fi

gpu_flag="--use_gpu"
if ! is_truthy "${USE_GPU}"; then
    gpu_flag="--no-use_gpu"
fi

container_cmd=$(cat <<EOF
set -euo pipefail
cd $(printf '%q' "${MODEL_ROOT}")
python $(printf '%q' "${COMMON_PY}") --video $(printf '%q' "${VIDEO_PATH}") --output-dir $(printf '%q' "${TEMP_FRAME_ROOT}")
python main.py \
  --mode stride-vipe \
  --stride_backend hmp \
  --video $(printf '%q' "${VIDEO_STEM}") \
  --image_folder $(printf '%q' "${TEMP_FRAME_ROOT}") \
  --wilor_cache_root $(printf '%q' "${WILOR_CACHE_ROOT}") \
  --stride_output_folder $(printf '%q' "${STRIDE_OUTPUT_ROOT}") \
  --stride_from_cache \
  --vipe_output_root $(printf '%q' "${VIPE_OUTPUT_ROOT}") \
  --hmp_assets_root $(printf '%q' "${HMP_ASSETS_ROOT}") \
  --mano_model_path $(printf '%q' "${MANO_MODEL_PATH}") \
  --target_hand $(printf '%q' "${TARGET_HAND}") \
  ${visualize_flag} \
  ${gpu_flag}
EOF
)

echo "STRIDE video: ${VIDEO_PATH}"
echo "WiLoR cache root: ${WILOR_CACHE_ROOT}"
echo "STRIDE output root: ${STRIDE_OUTPUT_ROOT}"
echo "STRIDE mode: stride-vipe"
echo "STRIDE backend: hmp"
echo "ViPE output root: ${VIPE_OUTPUT_ROOT}"
echo "HMP assets root: ${HMP_ASSETS_ROOT}"
echo "STRIDE completion marker: ${MARKER_PATH}"
echo "STRIDE temp frame dir: ${TEMP_FRAME_DIR}"

srun apptainer exec \
  --nv \
  --bind /scratch:/scratch \
  --bind "${TEMP_PARENT}:${TEMP_PARENT}" \
  "${APPTAINER_IMAGE}" \
  bash -lc "${container_cmd}"

rm -rf "${VIDEO_OUTPUT_DIR}/visualizations"
rm -f "${VIDEO_OUTPUT_FILE}"
write_simple_completion_marker "${MARKER_PATH}" "stride-vipe" "${VIDEO_STEM}" "${VIDEO_OUTPUT_DIR}"

echo "==============================================="
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
hours=$(printf "%.2f" "$(echo "$elapsed/3600" | bc -l)")

echo "Job finished at: $(date)"
echo "Execution took $hours hours"
echo "Writing output to $outfile"
echo "Retained STRIDE outputs: ${VIDEO_OUTPUT_DIR}"
echo "==============================================="
