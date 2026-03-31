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

#SBATCH --job-name=wilor-train
#SBATCH --partition=gpu-a100-small
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=8G
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

# ================ MODULES ================

module load 2024r1
module load cuda/11.7
module load python

# ================ HELPERS ================

is_true() {
  local value="${1:-false}"
  shopt -s nocasematch
  case "$value" in
    1|true|yes|y|on) shopt -u nocasematch; return 0 ;;
    *) shopt -u nocasematch; return 1 ;;
  esac
}

append_bool_flag() {
  local flag_name="$1"
  local flag_value="$2"
  if is_true "$flag_value"; then
    PYTHON_CMD+=("--${flag_name}")
  else
    PYTHON_CMD+=("--no-${flag_name}")
  fi
}

print_command() {
  printf 'Command:'
  printf ' %q' "$@"
  printf '\n'
}

# ================ JOB INFO ================

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

# ================ PATHS ================

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/mthakur/manifold}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/models/wilor_hands}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/whim/training}"
VIPE_OUTPUT_ROOT="${VIPE_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/vipe}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/wilor_finetune}"

CHECKPOINT="${CHECKPOINT:-${MODEL_ROOT}/pretrained_models/wilor_final.ckpt}"
CFG_PATH="${CFG_PATH:-${MODEL_ROOT}/pretrained_models/model_config.yaml}"
DETECTOR_PATH="${DETECTOR_PATH:-${MODEL_ROOT}/pretrained_models/detector.pt}"
POSE_DIR="${POSE_DIR:-${VIPE_OUTPUT_ROOT}/pose}"
INTRINSICS_DIR="${INTRINSICS_DIR:-${VIPE_OUTPUT_ROOT}/intrinsics}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-${MODEL_ROOT}/apptainer/template.sif}"

# ================ TRAINING CONFIG ================

TRAIN_MODE="${TRAIN_MODE:-distill}"        # distill | supervised
TRAIN_SCOPE="${TRAIN_SCOPE:-refine_net}"   # camera_head | refine_net | full
CAMERA_LOSS_WEIGHT="${CAMERA_LOSS_WEIGHT:-0.01}"
LR="${LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-2000}"
LOG_EVERY="${LOG_EVERY:-25}"
SAVE_EVERY="${SAVE_EVERY:-250}"
SEED="${SEED:-42}"
RESCALE_FACTOR="${RESCALE_FACTOR:-2.0}"
USE_GPU="${USE_GPU:-true}"
AMP="${AMP:-true}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"

# Distillation-specific inputs
IMAGE_FOLDER="${IMAGE_FOLDER:-${DATA_ROOT}/images}"
VIDEO_NAME="${VIDEO_NAME:-clip_2}"
ALL_VIDEOS="${ALL_VIDEOS:-false}"
DETECTION_CONF="${DETECTION_CONF:-0.3}"
DETECTION_CACHE="${DETECTION_CACHE:-}"

# Supervised-specific inputs
DATASET_FILE="${DATASET_FILE:-}"
IMG_DIR="${IMG_DIR:-}"
VIDEO_NAME_OVERRIDE="${VIDEO_NAME_OVERRIDE:-}"
FRAME_INDEX_PATTERN="${FRAME_INDEX_PATTERN:-"(\\d+)$"}"
SHUFFLE="${SHUFFLE:-true}"
MOCAP_FILE="${MOCAP_FILE:-}"
ADVERSARIAL_WEIGHT="${ADVERSARIAL_WEIGHT:-0.0}"

if [[ -z "${RUN_NAME:-}" ]]; then
  if [[ "$TRAIN_MODE" == "distill" ]]; then
    if is_true "$ALL_VIDEOS"; then
      RUN_NAME="distill_all_videos"
    else
      RUN_NAME="distill_${VIDEO_NAME}"
    fi
  else
    if [[ -n "$DATASET_FILE" ]]; then
      RUN_NAME="supervised_$(basename "${DATASET_FILE%.*}")"
    else
      RUN_NAME="supervised_run"
    fi
  fi
fi
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"

mkdir -p "${OUTPUT_ROOT}" "${RUN_OUTPUT_DIR}"

if [[ ! -f "${APPTAINER_IMAGE}" ]]; then
  echo "Apptainer image not found: ${APPTAINER_IMAGE}" >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

if [[ ! -f "${CFG_PATH}" ]]; then
  echo "Model config not found: ${CFG_PATH}" >&2
  exit 1
fi

# ================ COMMAND BUILD ================

case "$TRAIN_MODE" in
  distill)
    if [[ -z "$DETECTION_CACHE" ]]; then
      if is_true "$ALL_VIDEOS"; then
        DETECTION_CACHE="${RUN_OUTPUT_DIR}/detections_all_videos.json"
      else
        DETECTION_CACHE="${RUN_OUTPUT_DIR}/detections_${VIDEO_NAME}.json"
      fi
    fi
    mkdir -p "$(dirname "${DETECTION_CACHE}")"

    PYTHON_CMD=(
      python -u "${MODEL_ROOT}/finetune_wilor_distill_vipe.py"
      --checkpoint "${CHECKPOINT}"
      --cfg_path "${CFG_PATH}"
      --detector_path "${DETECTOR_PATH}"
      --image_folder "${IMAGE_FOLDER}"
      --pose_dir "${POSE_DIR}"
      --intrinsics_dir "${INTRINSICS_DIR}"
      --output_dir "${RUN_OUTPUT_DIR}"
      --train_scope "${TRAIN_SCOPE}"
      --camera_loss_weight "${CAMERA_LOSS_WEIGHT}"
      --lr "${LR}"
      --weight_decay "${WEIGHT_DECAY}"
      --batch_size "${BATCH_SIZE}"
      --num_workers "${NUM_WORKERS}"
      --max_steps "${MAX_STEPS}"
      --log_every "${LOG_EVERY}"
      --save_every "${SAVE_EVERY}"
      --seed "${SEED}"
      --rescale_factor "${RESCALE_FACTOR}"
      --detection_conf "${DETECTION_CONF}"
    )

    if [[ "$SAMPLE_LIMIT" != "0" ]]; then
      PYTHON_CMD+=(--sample_limit "${SAMPLE_LIMIT}")
    fi

    PYTHON_CMD+=(--detection_cache "${DETECTION_CACHE}")

    if is_true "$ALL_VIDEOS"; then
      PYTHON_CMD+=(--all_videos)
    else
      PYTHON_CMD+=(--video "${VIDEO_NAME}")
    fi
    ;;

  supervised)
    if [[ -z "$DATASET_FILE" ]]; then
      echo "DATASET_FILE must be set when TRAIN_MODE=supervised." >&2
      exit 1
    fi

    if [[ -z "$IMG_DIR" ]]; then
      echo "IMG_DIR must be set when TRAIN_MODE=supervised." >&2
      exit 1
    fi

    PYTHON_CMD=(
      python -u "${MODEL_ROOT}/finetune_wilor_supervised.py"
      --checkpoint "${CHECKPOINT}"
      --cfg_path "${CFG_PATH}"
      --dataset_file "${DATASET_FILE}"
      --img_dir "${IMG_DIR}"
      --pose_dir "${POSE_DIR}"
      --intrinsics_dir "${INTRINSICS_DIR}"
      --output_dir "${RUN_OUTPUT_DIR}"
      --frame_index_pattern "${FRAME_INDEX_PATTERN}"
      --rescale_factor "${RESCALE_FACTOR}"
      --train_scope "${TRAIN_SCOPE}"
      --camera_loss_weight "${CAMERA_LOSS_WEIGHT}"
      --lr "${LR}"
      --weight_decay "${WEIGHT_DECAY}"
      --batch_size "${BATCH_SIZE}"
      --num_workers "${NUM_WORKERS}"
      --max_steps "${MAX_STEPS}"
      --log_every "${LOG_EVERY}"
      --save_every "${SAVE_EVERY}"
      --seed "${SEED}"
      --adversarial_weight "${ADVERSARIAL_WEIGHT}"
    )

    if [[ "$SAMPLE_LIMIT" != "0" ]]; then
      PYTHON_CMD+=(--sample_limit "${SAMPLE_LIMIT}")
    fi

    if [[ -n "$VIDEO_NAME_OVERRIDE" ]]; then
      PYTHON_CMD+=(--video_name "${VIDEO_NAME_OVERRIDE}")
    fi

    if [[ -n "$MOCAP_FILE" ]]; then
      PYTHON_CMD+=(--mocap_file "${MOCAP_FILE}")
    fi
    ;;

  *)
    echo "Unsupported TRAIN_MODE='${TRAIN_MODE}'. Use 'distill' or 'supervised'." >&2
    exit 1
    ;;
esac

append_bool_flag "use_gpu" "${USE_GPU}"
append_bool_flag "amp" "${AMP}"

if [[ "$TRAIN_MODE" == "supervised" ]]; then
  append_bool_flag "shuffle" "${SHUFFLE}"
fi

echo "Project root:    ${PROJECT_ROOT}"
echo "Model root:      ${MODEL_ROOT}"
echo "Output dir:      ${RUN_OUTPUT_DIR}"
echo "Train mode:      ${TRAIN_MODE}"
echo "Train scope:     ${TRAIN_SCOPE}"
echo "Checkpoint:      ${CHECKPOINT}"
echo "Pose dir:        ${POSE_DIR}"
echo "Intrinsics dir:  ${INTRINSICS_DIR}"
print_command "${PYTHON_CMD[@]}"

# ================ EXECUTION ================

APPTAINER_ARGS=(
  exec
  --nv
  --bind /scratch:/scratch
)

if [[ -d "${HOME}/.cache/torch" ]]; then
  APPTAINER_ARGS+=(--bind "${HOME}/.cache/torch:/home/mthakur/.cache/torch")
fi

if [[ -d "${HOME}/.cache/huggingface" ]]; then
  APPTAINER_ARGS+=(--bind "${HOME}/.cache/huggingface:/home/mthakur/.cache/huggingface")
fi

srun apptainer "${APPTAINER_ARGS[@]}" "${APPTAINER_IMAGE}" "${PYTHON_CMD[@]}"

echo "==============================================="
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
hours=$(printf "%.2f" "$(echo "$elapsed/3600" | bc -l)")

echo "Job finished at: $(date)"
echo "Execution took $hours hours"
echo "Writing output to $outfile"
echo "==============================================="
