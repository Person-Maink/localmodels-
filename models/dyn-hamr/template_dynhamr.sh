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

#SBATCH --job-name=dynhamr-inference
#SBATCH --partition=gpu-a100
#SBATCH --time=__TIME__
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=24G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=%x.out
set -euo pipefail

export HYDRA_FULL_ERROR=1
export LD_LIBRARY_PATH=/cm/local/apps/gcc/10.2.0/lib64/libstdc++.so.6.0.28:$LD_LIBRARY_PATH

base_name="${SLURM_JOB_NAME}"
dir="SLURM_logs"
mkdir -p "$dir"
count=$(printf "%03d" $(($(ls "$dir" 2>/dev/null | grep -c "^${base_name}_[0-9]\+\.out$") + 1)))
outfile="${dir}/${base_name}_${count}.out"
exec >"$outfile" 2>&1

module load 2024r1
module load cuda/11.7

PROJECT_ROOT="/scratch/mthakur/manifold"
MODEL_ROOT="${PROJECT_ROOT}/models/dyn-hamr"
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
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/dynhamr}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/data/images}"
VIDEO_NAME="${VIDEO_NAME:-__NAME__}"
VIDEO_EXT="${VIDEO_EXT:-__VIDEO_EXT__}"
VIDEO_FILE="${VIDEO_FILE:-${VIDEO_NAME}.${VIDEO_EXT}}"
IS_STATIC="${IS_STATIC:-false}"
RUN_PRIOR="${RUN_PRIOR:-false}"
RUN_VIS="${RUN_VIS:-false}"
TEMPORAL_SMOOTH="${TEMPORAL_SMOOTH:-false}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:--1}"
CHUNK_SECONDS="${CHUNK_SECONDS:-600}"
SKIP_EXISTING_CHUNKS="${SKIP_EXISTING_CHUNKS:-true}"
OVERWRITE="${OVERWRITE:-false}"
KEEP_TEMP_FRAMES="${KEEP_TEMP_FRAMES:-false}"
ROOT_ITERS="${ROOT_ITERS:-40}"
SMOOTH_ITERS="${SMOOTH_ITERS:-60}"
TEMP_PARENT="${TEMP_ROOT:-$(pick_temp_root)}"
DETECTRON2_CKPT="${DETECTRON2_CKPT:-${MODEL_ASSETS_ROOT}/common/detectron2/model_final_f05665.pkl}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-${MODEL_ROOT}/apptainer/template.sif}"

if ! VIDEO_PATH=$(resolve_video_path "${VIDEO_DIR}" "${VIDEO_NAME}" "${VIDEO_FILE}"); then
  echo "Could not resolve a supported video under ${VIDEO_DIR} for VIDEO_NAME='${VIDEO_NAME}' VIDEO_FILE='${VIDEO_FILE}'" >&2
  exit 1
fi

VIDEO_PATH=$(cd "$(dirname "${VIDEO_PATH}")" && pwd)/$(basename "${VIDEO_PATH}")
VIDEO_STEM="$(basename "${VIDEO_PATH%.*}")"
MARKER_PATH="$(completion_marker_path "${OUTPUT_ROOT}" "${VIDEO_STEM}")"

if [[ ! -f "${DETECTRON2_CKPT}" ]]; then
  echo "Detectron2 checkpoint not found: ${DETECTRON2_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${APPTAINER_IMAGE}" ]]; then
  echo "Apptainer image not found: ${APPTAINER_IMAGE}" >&2
  exit 1
fi

if [[ -f "${MARKER_PATH}" ]] && ! is_truthy "${OVERWRITE}"; then
  echo "Skipping ${VIDEO_STEM}: completion marker already exists at ${MARKER_PATH}"
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${TEMP_PARENT}"

if is_truthy "${RUN_VIS}"; then
  echo "Dyn-HaMR mode: optimization + visualization"
else
  echo "Dyn-HaMR mode: optimization only"
  echo "Visualization outputs are disabled by default. Set RUN_VIS=true to restore rendered outputs."
fi
echo "Dyn-HaMR artifacts root: ${LOG_ROOT}"
echo "Dyn-HaMR completion marker: ${MARKER_PATH}"
echo "Dyn-HaMR temp parent: ${TEMP_PARENT}"

export APPTAINERENV_HAMER_DETECTRON2_CKPT="${DETECTRON2_CKPT}"
export APPTAINERENV_MODEL_ASSETS_ROOT="${MODEL_ASSETS_ROOT}"

cmd=(
  python -u "${MODEL_ROOT}/run_chunked.py"
  --video "${VIDEO_PATH}"
  --video-name "${VIDEO_STEM}"
  --video-ext "${VIDEO_EXT}"
  --log-root "${LOG_ROOT}"
  --temp-parent "${TEMP_PARENT}"
  --run-vis "${RUN_VIS}"
  --run-prior "${RUN_PRIOR}"
  --is-static "${IS_STATIC}"
  --temporal-smooth "${TEMPORAL_SMOOTH}"
  --start-idx "${START_IDX}"
  --end-idx "${END_IDX}"
  --chunk-seconds "${CHUNK_SECONDS}"
  --root-iters "${ROOT_ITERS}"
  --smooth-iters "${SMOOTH_ITERS}"
  --skip-existing "${SKIP_EXISTING_CHUNKS}"
  --overwrite "${OVERWRITE}"
)

if is_truthy "${KEEP_TEMP_FRAMES}"; then
  cmd+=(--keep-temp-data)
fi

srun apptainer exec \
  --nv \
  --bind /scratch:/scratch \
  --bind "${TEMP_PARENT}:${TEMP_PARENT}" \
  "${APPTAINER_IMAGE}" \
  "${cmd[@]}"

echo "==============================================="
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
hours=$(printf "%.2f" "$(echo "$elapsed/3600" | bc -l)")

echo "Job finished at: $(date)"
echo "Execution took $hours hours"
echo "Writing output to $outfile"
echo "Dyn-HaMR artifacts root: ${LOG_ROOT}"
echo "==============================================="
