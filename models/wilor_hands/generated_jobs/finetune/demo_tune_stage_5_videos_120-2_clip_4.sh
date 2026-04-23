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

#SBATCH --job-name=wilor-inference
#SBATCH --partition=gpu-a100-small
#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=10G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=%x.out
set -euo pipefail

# ================ OUTPUT FILES ================
base_name="${SLURM_JOB_NAME}"
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

VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/data/test}"
VIDEO_NAME="${VIDEO_NAME:-120-2_clip_4}"
VIDEO_FILE="${VIDEO_FILE:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-tune_stage_5_videos}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_ROOT}/outputs/wilor_finetune/${EXPERIMENT_NAME}/best.ckpt}"
CFG_PATH="${CFG_PATH:-${MODEL_ROOT}/pretrained_models/model_config.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/wilor_finetune/${EXPERIMENT_NAME}}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-${MODEL_ROOT}/apptainer/template.sif}"
OVERWRITE="${OVERWRITE:-false}"
KEEP_TEMP_FRAMES="${KEEP_TEMP_FRAMES:-false}"
VISUALIZE="${VISUALIZE:-false}"
SAVE_MESH="${SAVE_MESH:-true}"
USE_GPU="${USE_GPU:-true}"
TEMP_PARENT="${TEMP_ROOT:-$(pick_temp_root)}"

if ! VIDEO_PATH=$(resolve_video_path "${VIDEO_DIR}" "${VIDEO_NAME}" "${VIDEO_FILE}"); then
    echo "Could not resolve a supported video under ${VIDEO_DIR} for VIDEO_NAME='${VIDEO_NAME}' VIDEO_FILE='${VIDEO_FILE}'" >&2
    exit 1
fi

VIDEO_PATH=$(cd "$(dirname "${VIDEO_PATH}")" && pwd)/$(basename "${VIDEO_PATH}")
VIDEO_STEM="$(basename "${VIDEO_PATH%.*}")"
MARKER_PATH="$(completion_marker_path "${OUTPUT_ROOT}" "${VIDEO_STEM}")"
VIDEO_OUTPUT_DIR="${OUTPUT_ROOT}/${VIDEO_STEM}"
VIDEO_OUTPUT_FILE="${OUTPUT_ROOT}/videos/${VIDEO_STEM}.mp4"

if [[ ! -f "${APPTAINER_IMAGE}" ]]; then
    echo "Apptainer image not found: ${APPTAINER_IMAGE}" >&2
    exit 1
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "Finetuned checkpoint not found: ${CHECKPOINT_PATH}" >&2
    exit 1
fi

if [[ ! -f "${CFG_PATH}" ]]; then
    echo "Model config not found: ${CFG_PATH}" >&2
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

mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${TEMP_PARENT}"
TEMP_WORKDIR=$(mktemp -d "${TEMP_PARENT%/}/wilor_${VIDEO_STEM}.XXXXXX")
TEMP_FRAME_DIR="${TEMP_WORKDIR}/${VIDEO_STEM}_frames"

cleanup_temp() {
    if is_truthy "${KEEP_TEMP_FRAMES}"; then
        echo "Keeping temporary WiLoR frames at ${TEMP_WORKDIR}"
        return
    fi
    rm -rf "${TEMP_WORKDIR}"
}
trap cleanup_temp EXIT

visualize_flag="--no-visualize"
if is_truthy "${VISUALIZE}"; then
    visualize_flag="--visualize"
fi

save_mesh_flag="--save_mesh"
if ! is_truthy "${SAVE_MESH}"; then
    save_mesh_flag="--no-save_mesh"
fi

gpu_flag="--use_gpu"
if ! is_truthy "${USE_GPU}"; then
    gpu_flag="--no-use_gpu"
fi

container_cmd=$(cat <<EOF
set -euo pipefail
cd $(printf '%q' "${MODEL_ROOT}")
python $(printf '%q' "${COMMON_PY}") --video $(printf '%q' "${VIDEO_PATH}") --output-dir $(printf '%q' "${TEMP_FRAME_DIR}")
python main.py --image_folder $(printf '%q' "${TEMP_FRAME_DIR}") --output_folder $(printf '%q' "${OUTPUT_ROOT}") --checkpoint_path $(printf '%q' "${CHECKPOINT_PATH}") --cfg_path $(printf '%q' "${CFG_PATH}") ${visualize_flag} ${save_mesh_flag} ${gpu_flag}
EOF
)

echo "WiLoR finetune experiment: ${EXPERIMENT_NAME}"
echo "WiLoR checkpoint: ${CHECKPOINT_PATH}"
echo "WiLoR config: ${CFG_PATH}"
echo "WiLoR video: ${VIDEO_PATH}"
echo "WiLoR output root: ${OUTPUT_ROOT}"
echo "WiLoR completion marker: ${MARKER_PATH}"
echo "WiLoR temp frame dir: ${TEMP_FRAME_DIR}"

srun apptainer exec \
  --nv \
  --bind /scratch:/scratch \
  --bind "${TEMP_PARENT}:${TEMP_PARENT}" \
  "${APPTAINER_IMAGE}" \
  bash -lc "${container_cmd}"

rm -rf "${VIDEO_OUTPUT_DIR}/visualizations"
rm -f "${VIDEO_OUTPUT_FILE}"
write_simple_completion_marker "${MARKER_PATH}" "wilor" "${VIDEO_STEM}" "${VIDEO_OUTPUT_DIR}/meshes"

echo "==============================================="
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
hours=$(printf "%.2f" "$(echo "$elapsed/3600" | bc -l)")

echo "Job finished at: $(date)"
echo "Execution took $hours hours"
echo "Writing output to $outfile"
echo "Retained WiLoR outputs: ${VIDEO_OUTPUT_DIR}/meshes"
echo "==============================================="
