#!/bin/bash
set -euo pipefail

# ================ SLURM SETUP ================
# Available HPC Partitions:
#   compute / compute-p1   : CPU jobs (48 CPUs, 185 GB RAM, Phase 1)
#   compute-p2             : CPU jobs (64 CPUs, 250 GB RAM, Phase 2)
#   gpu / gpu-v100         : GPU jobs (4x V100, 32 GB VRAM each, Phase 1)
#   gpu-a100               : GPU jobs (4x A100, 80 GB VRAM each, Phase 2)
#   gpu-a100-small         : Small GPU jobs (≤1 GPU, ≤10 GB VRAM, ≤2 CPUs, ≤4h)
#   memory                 : High-memory CPU jobs (>250 GB RAM)
#   visual                 : Visualization jobs

#SBATCH --job-name=stride-inference
#SBATCH --partition=gpu-a100
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=24G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=%x.out

base_name="${SLURM_JOB_NAME}"
dir="SLURM_logs"
mkdir -p "$dir"
count=$(printf "%03d" $(($(ls "$dir" 2>/dev/null | grep -c "^${base_name}_[0-9]\+\.out$") + 1)))
outfile="${dir}/${base_name}_${count}.out"
exec >"$outfile" 2>&1

module load 2024r1
module load cuda/11.7

PROJECT_ROOT="/scratch/mthakur/manifold"
WILOR_ROOT="${PROJECT_ROOT}/models/wilor_hands"
COMMON_PY="${PROJECT_ROOT}/models/common/extract_video_frames.py"
COMMON_SH="${PROJECT_ROOT}/models/common/inference_common.sh"

source "${COMMON_SH}"

echo "Loaded modules:"
module list 2>&1

nvidia-smi

echo "================ SLURM JOB INFO ================"
echo "Job ID:         $SLURM_JOB_ID"
echo "Job Name:       $SLURM_JOB_NAME"
echo "Partition:      $SLURM_JOB_PARTITION"
echo "Node List:      $SLURM_JOB_NODELIST"
echo "CPUs per task:  $SLURM_CPUS_PER_TASK"
echo "GPUs per task:  $SLURM_GPUS_PER_TASK"
echo "Memory per GPU: $SLURM_MEM_PER_GPU"
echo "Submit dir:     $SLURM_SUBMIT_DIR"
echo "Work dir:       $(pwd)"
echo "Job started at: $(date)"
start_time=$(date +%s)
echo "==============================================="

MODEL_ASSETS_ROOT="${MODEL_ASSETS_ROOT:-${PROJECT_ROOT}/models/model_assets}"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/data/images}"
VIDEO_NAME="${VIDEO_NAME:-}"
VIDEO_FILE="${VIDEO_FILE:-}"
WILOR_OUTPUT_ROOT="${WILOR_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/wilor}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/stride}"
HMP_ASSETS_ROOT="${HMP_ASSETS_ROOT:-${WILOR_ROOT}/_DATA/hmp_model}"
MANO_MODEL_PATH="${MANO_MODEL_PATH:-${WILOR_ROOT}/mano_data}"
HMP_CONFIG_NAME="${HMP_CONFIG_NAME:-hmp_config.yaml}"
STRIDE_BACKEND="${STRIDE_BACKEND:-hmp}"
OVERWRITE="${OVERWRITE:-false}"
KEEP_TEMP_FRAMES="${KEEP_TEMP_FRAMES:-false}"
STRIDE_FROM_CACHE="${STRIDE_FROM_CACHE:-true}"
VISUALIZE="${VISUALIZE:-false}"
TARGET_HAND="${TARGET_HAND:-auto}"
STRIDE_ITERS="${STRIDE_ITERS:-300}"
STRIDE_LR="${STRIDE_LR:-0.05}"
STRIDE_OBS_WEIGHT="${STRIDE_OBS_WEIGHT:-10.0}"
STRIDE_REPROJ_WEIGHT="${STRIDE_REPROJ_WEIGHT:-2.0}"
STRIDE_SHAPE_WEIGHT="${STRIDE_SHAPE_WEIGHT:-5.0}"
STRIDE_CAM_SMOOTH_WEIGHT="${STRIDE_CAM_SMOOTH_WEIGHT:-25.0}"
STRIDE_POSE_SMOOTH_WEIGHT="${STRIDE_POSE_SMOOTH_WEIGHT:-1.5}"
STRIDE_JOINT_SMOOTH_WEIGHT="${STRIDE_JOINT_SMOOTH_WEIGHT:-0.5}"
STRIDE_ANCHOR_WEIGHT="${STRIDE_ANCHOR_WEIGHT:-0.5}"
STRIDE_FFT_WEIGHT="${STRIDE_FFT_WEIGHT:-0.0}"
STRIDE_FFT_BAND_LOW_HZ="${STRIDE_FFT_BAND_LOW_HZ:-}"
STRIDE_FFT_BAND_HIGH_HZ="${STRIDE_FFT_BAND_HIGH_HZ:-}"
STRIDE_FPS="${STRIDE_FPS:-30.0}"
STRIDE_POSE_RANK="${STRIDE_POSE_RANK:-32}"
STRIDE_CAM_RANK="${STRIDE_CAM_RANK:-16}"
TEMP_PARENT="${TEMP_ROOT:-$(pick_temp_root)}"
WILOR_IMAGE="${WILOR_IMAGE:-${WILOR_ROOT}/apptainer/template.sif}"

if [[ -z "${VIDEO_NAME}" && -z "${VIDEO_FILE}" ]]; then
    echo "Set VIDEO_NAME or VIDEO_FILE before running STRIDE inference." >&2
    exit 1
fi

if ! VIDEO_PATH=$(resolve_video_path "${VIDEO_DIR}" "${VIDEO_NAME}" "${VIDEO_FILE}"); then
    echo "Could not resolve a supported video under ${VIDEO_DIR} for VIDEO_NAME='${VIDEO_NAME}' VIDEO_FILE='${VIDEO_FILE}'" >&2
    exit 1
fi

VIDEO_PATH=$(cd "$(dirname "${VIDEO_PATH}")" && pwd)/$(basename "${VIDEO_PATH}")
VIDEO_STEM="$(basename "${VIDEO_PATH%.*}")"
MARKER_PATH="$(completion_marker_path "${OUTPUT_ROOT}" "${VIDEO_STEM}")"
WILOR_MARKER_PATH="$(completion_marker_path "${WILOR_OUTPUT_ROOT}" "${VIDEO_STEM}")"

for required_file in "${WILOR_IMAGE}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file not found: ${required_file}" >&2
        exit 1
    fi
done

if [[ "${STRIDE_BACKEND}" == "hmp" && ! -d "${HMP_ASSETS_ROOT}" ]]; then
    echo "HMP assets root not found: ${HMP_ASSETS_ROOT}" >&2
    exit 1
fi

if [[ -f "${MARKER_PATH}" ]] && ! is_truthy "${OVERWRITE}"; then
    echo "Skipping ${VIDEO_STEM}: completion marker already exists at ${MARKER_PATH}"
    exit 0
fi

mkdir -p "${WILOR_OUTPUT_ROOT}" "${OUTPUT_ROOT}" "${TEMP_PARENT}"

TEMP_WORKDIR=$(mktemp -d "${TEMP_PARENT%/}/stride_${VIDEO_STEM}.XXXXXX")
TEMP_FRAME_DIR="${TEMP_WORKDIR}/${VIDEO_STEM}_frames"

cleanup_temp() {
    if is_truthy "${KEEP_TEMP_FRAMES}"; then
        echo "Keeping temporary STRIDE frames at ${TEMP_WORKDIR}"
        return
    fi
    rm -rf "${TEMP_WORKDIR}"
}
trap cleanup_temp EXIT

if is_truthy "${OVERWRITE}"; then
    rm -f "${MARKER_PATH}"
fi

if is_truthy "${STRIDE_FROM_CACHE}" && [[ ! -d "${WILOR_OUTPUT_ROOT}/${VIDEO_STEM}/meshes" ]]; then
    echo "Expected cached WiLoR meshes at ${WILOR_OUTPUT_ROOT}/${VIDEO_STEM}/meshes" >&2
    exit 1
fi

visualize_flag="--no-visualize"
if is_truthy "${VISUALIZE}"; then
    visualize_flag="--visualize"
fi

overwrite_flag="--no-overwrite"
if is_truthy "${OVERWRITE}"; then
    overwrite_flag="--overwrite"
fi

stride_from_cache_flag="--no-stride_from_cache"
if is_truthy "${STRIDE_FROM_CACHE}"; then
    stride_from_cache_flag="--stride_from_cache"
fi

fft_low_flag=""
if [[ -n "${STRIDE_FFT_BAND_LOW_HZ}" ]]; then
    fft_low_flag="--stride_fft_band_low_hz $(printf '%q' "${STRIDE_FFT_BAND_LOW_HZ}")"
fi

fft_high_flag=""
if [[ -n "${STRIDE_FFT_BAND_HIGH_HZ}" ]]; then
    fft_high_flag="--stride_fft_band_high_hz $(printf '%q' "${STRIDE_FFT_BAND_HIGH_HZ}")"
fi

container_cmd=$(cat <<EOF
set -euo pipefail
cd $(printf '%q' "${WILOR_ROOT}")
python main.py --mode stride --stride_backend $(printf '%q' "${STRIDE_BACKEND}") --image_folder $(printf '%q' "${TEMP_FRAME_DIR}") --output_folder $(printf '%q' "${WILOR_OUTPUT_ROOT}") --wilor_cache_root $(printf '%q' "${WILOR_OUTPUT_ROOT}") --stride_output_folder $(printf '%q' "${OUTPUT_ROOT}") --video $(printf '%q' "${VIDEO_STEM}") --target_hand $(printf '%q' "${TARGET_HAND}") --mano_model_path $(printf '%q' "${MANO_MODEL_PATH}") --hmp_assets_root $(printf '%q' "${HMP_ASSETS_ROOT}") --hmp_config_name $(printf '%q' "${HMP_CONFIG_NAME}") ${overwrite_flag} ${stride_from_cache_flag} ${visualize_flag} --save_mesh --use_gpu --stride_iters $(printf '%q' "${STRIDE_ITERS}") --stride_lr $(printf '%q' "${STRIDE_LR}") --stride_obs_weight $(printf '%q' "${STRIDE_OBS_WEIGHT}") --stride_reproj_weight $(printf '%q' "${STRIDE_REPROJ_WEIGHT}") --stride_shape_weight $(printf '%q' "${STRIDE_SHAPE_WEIGHT}") --stride_cam_smooth_weight $(printf '%q' "${STRIDE_CAM_SMOOTH_WEIGHT}") --stride_pose_smooth_weight $(printf '%q' "${STRIDE_POSE_SMOOTH_WEIGHT}") --stride_joint_smooth_weight $(printf '%q' "${STRIDE_JOINT_SMOOTH_WEIGHT}") --stride_anchor_weight $(printf '%q' "${STRIDE_ANCHOR_WEIGHT}") --stride_fft_weight $(printf '%q' "${STRIDE_FFT_WEIGHT}") --stride_fps $(printf '%q' "${STRIDE_FPS}") --stride_pose_rank $(printf '%q' "${STRIDE_POSE_RANK}") --stride_cam_rank $(printf '%q' "${STRIDE_CAM_RANK}") ${fft_low_flag} ${fft_high_flag}
EOF
)

if ! is_truthy "${STRIDE_FROM_CACHE}"; then
    container_cmd=$(cat <<EOF
set -euo pipefail
cd $(printf '%q' "${WILOR_ROOT}")
python $(printf '%q' "${COMMON_PY}") --video $(printf '%q' "${VIDEO_PATH}") --output-dir $(printf '%q' "${TEMP_FRAME_DIR}")
python main.py --mode stride --stride_backend $(printf '%q' "${STRIDE_BACKEND}") --image_folder $(printf '%q' "${TEMP_FRAME_DIR}") --output_folder $(printf '%q' "${WILOR_OUTPUT_ROOT}") --wilor_cache_root $(printf '%q' "${WILOR_OUTPUT_ROOT}") --stride_output_folder $(printf '%q' "${OUTPUT_ROOT}") --video $(printf '%q' "${VIDEO_STEM}") --target_hand $(printf '%q' "${TARGET_HAND}") --mano_model_path $(printf '%q' "${MANO_MODEL_PATH}") --hmp_assets_root $(printf '%q' "${HMP_ASSETS_ROOT}") --hmp_config_name $(printf '%q' "${HMP_CONFIG_NAME}") ${overwrite_flag} ${stride_from_cache_flag} ${visualize_flag} --save_mesh --use_gpu --stride_iters $(printf '%q' "${STRIDE_ITERS}") --stride_lr $(printf '%q' "${STRIDE_LR}") --stride_obs_weight $(printf '%q' "${STRIDE_OBS_WEIGHT}") --stride_reproj_weight $(printf '%q' "${STRIDE_REPROJ_WEIGHT}") --stride_shape_weight $(printf '%q' "${STRIDE_SHAPE_WEIGHT}") --stride_cam_smooth_weight $(printf '%q' "${STRIDE_CAM_SMOOTH_WEIGHT}") --stride_pose_smooth_weight $(printf '%q' "${STRIDE_POSE_SMOOTH_WEIGHT}") --stride_joint_smooth_weight $(printf '%q' "${STRIDE_JOINT_SMOOTH_WEIGHT}") --stride_anchor_weight $(printf '%q' "${STRIDE_ANCHOR_WEIGHT}") --stride_fft_weight $(printf '%q' "${STRIDE_FFT_WEIGHT}") --stride_fps $(printf '%q' "${STRIDE_FPS}") --stride_pose_rank $(printf '%q' "${STRIDE_POSE_RANK}") --stride_cam_rank $(printf '%q' "${STRIDE_CAM_RANK}") ${fft_low_flag} ${fft_high_flag}
EOF
)
fi

echo "STRIDE video: ${VIDEO_PATH}"
echo "WiLoR cache root: ${WILOR_OUTPUT_ROOT}"
echo "STRIDE output root: ${OUTPUT_ROOT}"
echo "STRIDE backend: ${STRIDE_BACKEND}"
echo "HMP assets root: ${HMP_ASSETS_ROOT}"
echo "STRIDE completion marker: ${MARKER_PATH}"

srun apptainer exec \
  --nv \
  --bind /scratch:/scratch \
  --bind "${TEMP_PARENT}:${TEMP_PARENT}" \
  "${WILOR_IMAGE}" \
  bash -lc "${container_cmd}"

if [[ -f "${WILOR_MARKER_PATH}" ]] || ! is_truthy "${STRIDE_FROM_CACHE}"; then
    write_simple_completion_marker "${WILOR_MARKER_PATH}" "wilor" "${VIDEO_STEM}" "${WILOR_OUTPUT_ROOT}/${VIDEO_STEM}/meshes"
fi

write_simple_completion_marker "${MARKER_PATH}" "stride" "${VIDEO_STEM}" "${OUTPUT_ROOT}/${VIDEO_STEM}"

echo "==============================================="
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
hours=$(printf "%.2f" "$(echo "$elapsed/3600" | bc -l)")

echo "Job finished at: $(date)"
echo "Execution took $hours hours"
echo "Writing output to $outfile"
echo "Retained STRIDE outputs: ${OUTPUT_ROOT}/${VIDEO_STEM}"
echo "==============================================="
