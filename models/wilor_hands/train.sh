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
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=16G
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

append_value_flag_if_set() {
  local flag_name="$1"
  local flag_value="${2:-}"
  if [[ -n "$flag_value" ]]; then
    PYTHON_CMD+=("$flag_name" "$flag_value")
  fi
}

append_optional_bool_override() {
  local flag_name="$1"
  local flag_value="${2:-}"
  if [[ -z "$flag_value" ]]; then
    return
  fi
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

set_from_config_if_unset() {
  local var_name="$1"
  local encoded_value="$2"
  local decoded_value=""
  if [[ -n "${!var_name+x}" ]]; then
    return 0
  fi
  decoded_value="$(printf '%s' "$encoded_value" | base64 --decode)"
  printf -v "$var_name" '%s' "$decoded_value"
}

resolve_video_path() {
  local image_root="$1"
  local video_name="$2"
  local candidate=""

  shopt -s nullglob
  for candidate in \
    "${image_root}/${video_name}.mp4" \
    "${image_root}/${video_name}.MP4" \
    "${image_root}/${video_name}.avi" \
    "${image_root}/${video_name}.AVI" \
    "${image_root}/${video_name}.mts" \
    "${image_root}/${video_name}.MTS" \
    "${image_root}/${video_name}.mov" \
    "${image_root}/${video_name}.MOV"; do
    [[ -f "$candidate" ]] || continue
    printf '%s' "$candidate"
    shopt -u nullglob
    return 0
  done
  shopt -u nullglob
  return 1
}

list_candidate_videos() {
  local image_root="$1"
  local entry=""
  local video_name=""
  local -a candidates=()

  shopt -s nullglob
  for entry in "${image_root}"/*_frames; do
    [[ -d "$entry" ]] || continue
    video_name="$(basename "$entry")"
    candidates+=("${video_name%_frames}")
  done

  for entry in "${image_root}"/*; do
    [[ -f "$entry" ]] || continue
    case "${entry##*.}" in
      mp4|MP4|avi|AVI|mts|MTS|mov|MOV)
        video_name="$(basename "$entry")"
        candidates+=("${video_name%.*}")
        ;;
    esac
  done
  shopt -u nullglob

  if (( ${#candidates[@]} == 0 )); then
    return 0
  fi

  printf '%s\n' "${candidates[@]}" | awk '!seen[$0]++'
}

pick_random_eligible_distill_video() {
  local image_root="$1"
  local pose_dir="$2"
  local intrinsics_dir="$3"
  local requested_video="${4:-}"
  local requested_videos="${5:-}"
  local found_requested="false"
  local video_name=""
  local allowed_requested="false"
  local -A requested_subset=()
  local -a eligible_videos=()

  if [[ -n "$requested_videos" ]]; then
    local requested_parts=()
    IFS='|' read -r -a requested_parts <<< "$requested_videos"
    for video_name in "${requested_parts[@]}"; do
      [[ -n "$video_name" ]] || continue
      requested_subset["$video_name"]=1
    done
  fi

  if [[ -n "$requested_video" ]]; then
    while IFS= read -r video_name; do
      if [[ "$video_name" == "$requested_video" ]]; then
        found_requested="true"
        break
      fi
    done < <(list_candidate_videos "$image_root")
    if [[ "$found_requested" != "true" ]]; then
      echo "Requested VIDEO_NAME was not found under ${image_root}: ${requested_video}" >&2
      return 1
    fi
    if [[ ! -f "${pose_dir}/${requested_video}.npz" ]]; then
      echo "Requested VIDEO_NAME is missing a ViPE pose file: ${pose_dir}/${requested_video}.npz" >&2
      return 1
    fi
    if [[ ! -f "${intrinsics_dir}/${requested_video}.npz" ]]; then
      echo "Requested VIDEO_NAME is missing a ViPE intrinsics file: ${intrinsics_dir}/${requested_video}.npz" >&2
      return 1
    fi
    printf '%s' "$requested_video"
    return 0
  fi

  while IFS= read -r video_name; do
    [[ -z "$video_name" ]] && continue
    if (( ${#requested_subset[@]} > 0 )) && [[ -z "${requested_subset[$video_name]+x}" ]]; then
      continue
    fi
    if [[ -f "${pose_dir}/${video_name}.npz" && -f "${intrinsics_dir}/${video_name}.npz" ]]; then
      eligible_videos+=("$video_name")
      allowed_requested="true"
    fi
  done < <(list_candidate_videos "$image_root")

  if (( ${#requested_subset[@]} > 0 )) && [[ "$allowed_requested" != "true" ]]; then
    echo "No eligible requested videos were found under ${image_root} with matching ViPE artifacts." >&2
    return 1
  fi

  if (( ${#eligible_videos[@]} == 0 )); then
    echo "No eligible videos were found under ${image_root} with matching ViPE pose and intrinsics files." >&2
    return 1
  fi

  printf '%s\n' "${eligible_videos[@]}" | shuf -n 1
}

ensure_temp_image_root() {
  if [[ -n "${TEMP_IMAGE_ROOT:-}" ]]; then
    return 0
  fi

  mkdir -p "${TEMP_PARENT}"
  TEMP_WORKDIR=$(mktemp -d "${TEMP_PARENT%/}/wilor_train_${SLURM_JOB_ID:-manual}.XXXXXX")
  TEMP_IMAGE_ROOT="${TEMP_WORKDIR}/images"
  mkdir -p "${TEMP_IMAGE_ROOT}"
  echo "[progress] Created temporary training image root: ${TEMP_IMAGE_ROOT}"
}

cleanup_temp_frames() {
  if [[ -z "${TEMP_WORKDIR:-}" ]]; then
    return 0
  fi

  if is_true "${KEEP_TEMP_FRAMES:-false}"; then
    echo "Keeping temporary training frames at ${TEMP_WORKDIR}"
    return 0
  fi

  rm -rf "${TEMP_WORKDIR}"
}

stage_video_frames_for_finetune() {
  local source_image_root="$1"
  local video_name="$2"
  local source_frame_dir="${source_image_root}/${video_name}_frames"
  local target_frame_dir="${TEMP_IMAGE_ROOT}/${video_name}_frames"
  local video_path=""

  [[ -n "$video_name" ]] || return 0

  if [[ -e "${target_frame_dir}" ]]; then
    return 0
  fi

  if [[ -d "${source_frame_dir}" ]]; then
    ln -s "${source_frame_dir}" "${target_frame_dir}"
    echo "[progress] Reusing existing frames for ${video_name} via ${source_frame_dir}"
    return 0
  fi

  if ! video_path="$(resolve_video_path "${source_image_root}" "${video_name}")"; then
    echo "Could not find extracted frames or a supported raw video for ${video_name} under ${source_image_root}" >&2
    return 1
  fi

  echo "[progress] Extracting frames for ${video_name} into ${target_frame_dir}"
  apptainer exec \
    --nv \
    --bind /scratch:/scratch \
    --bind "${TEMP_PARENT}:${TEMP_PARENT}" \
    "${APPTAINER_IMAGE}" \
    python -u "${COMMON_PY}" --video "${video_path}" --output-dir "${target_frame_dir}"
}

prepare_image_root_for_finetune() {
  local source_image_root="$1"
  shift
  local video_name=""
  local -A seen_videos=()

  ensure_temp_image_root

  for video_name in "$@"; do
    [[ -n "$video_name" ]] || continue
    if [[ -n "${seen_videos[$video_name]+x}" ]]; then
      continue
    fi
    seen_videos["$video_name"]=1
    stage_video_frames_for_finetune "${source_image_root}" "${video_name}"
  done
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
COMMON_PY="${COMMON_PY:-${PROJECT_ROOT}/models/common/extract_video_frames.py}"
TEMP_PARENT="${TEMP_PARENT:-/tmp}"
KEEP_TEMP_FRAMES="${KEEP_TEMP_FRAMES:-false}"
TEMP_WORKDIR=""
TEMP_IMAGE_ROOT=""
SOURCE_IMAGE_FOLDER=""

trap cleanup_temp_frames EXIT

# ================ TRAINING CONFIG ================

LOSS_CONFIG="${LOSS_CONFIG:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"

if [[ -n "${LOSS_CONFIG}" ]]; then
  RESOLVE_CMD=(
    python -u "${MODEL_ROOT}/experiment_config.py"
    resolve
    --loss-config "${LOSS_CONFIG}"
    --format shell
  )
  if [[ -n "${EXPERIMENT_NAME}" ]]; then
    RESOLVE_CMD+=(--experiment-name "${EXPERIMENT_NAME}")
  fi
  while IFS=$'\t' read -r resolved_name resolved_value; do
    [[ -n "${resolved_name}" ]] || continue
    set_from_config_if_unset "${resolved_name}" "${resolved_value}"
  done < <("${RESOLVE_CMD[@]}")
fi

TRAIN_MODE="${TRAIN_MODE:-distill}"        # distill | test
TRAIN_SCOPE="${TRAIN_SCOPE:-refine_net}"   # camera_head | refine_net | full
CAMERA_LOSS_WEIGHT="${CAMERA_LOSS_WEIGHT:-0.01}"
VIPE_CAMERA_ENABLED="${VIPE_CAMERA_ENABLED:-}"
VIPE_CAMERA_WEIGHT="${VIPE_CAMERA_WEIGHT:-}"
TEMPORAL_CAMERA_ENABLED="${TEMPORAL_CAMERA_ENABLED:-}"
TEMPORAL_CAMERA_FORMULATION="${TEMPORAL_CAMERA_FORMULATION:-}"
TEMPORAL_CAMERA_WEIGHT="${TEMPORAL_CAMERA_WEIGHT:-}"
TEMPORAL_CAMERA_SCORER_WEIGHT="${TEMPORAL_CAMERA_SCORER_WEIGHT:-}"
TEMPORAL_BBOX_PROJECTED_ENABLED="${TEMPORAL_BBOX_PROJECTED_ENABLED:-}"
TEMPORAL_BBOX_PROJECTED_FORMULATION="${TEMPORAL_BBOX_PROJECTED_FORMULATION:-}"
TEMPORAL_BBOX_PROJECTED_WEIGHT="${TEMPORAL_BBOX_PROJECTED_WEIGHT:-}"
TEMPORAL_BBOX_PROJECTED_SCORER_WEIGHT="${TEMPORAL_BBOX_PROJECTED_SCORER_WEIGHT:-}"
TEMPORAL_BBOX_INPUT_ENABLED="${TEMPORAL_BBOX_INPUT_ENABLED:-}"
TEMPORAL_BBOX_INPUT_FORMULATION="${TEMPORAL_BBOX_INPUT_FORMULATION:-}"
TEMPORAL_BBOX_INPUT_WEIGHT="${TEMPORAL_BBOX_INPUT_WEIGHT:-}"
TEMPORAL_BBOX_INPUT_SCORER_WEIGHT="${TEMPORAL_BBOX_INPUT_SCORER_WEIGHT:-}"
LR="${LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-}"
LOG_EVERY="${LOG_EVERY:-}"
SAVE_EVERY="${SAVE_EVERY:-}"
SEED="${SEED:-42}"
RESCALE_FACTOR="${RESCALE_FACTOR:-2.0}"
USE_GPU="${USE_GPU:-true}"
AMP="${AMP:-true}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"
VALIDATION_SPLIT="${VALIDATION_SPLIT:-0.15}"
TEMPORAL_WINDOW_SIZE="${TEMPORAL_WINDOW_SIZE:-}"
TEMPORAL_WINDOW_STRIDE="${TEMPORAL_WINDOW_STRIDE:-}"
TEMPORAL_MAX_FRAME_GAP="${TEMPORAL_MAX_FRAME_GAP:-}"
TEMPORAL_REDUCTION="${TEMPORAL_REDUCTION:-}"
TEMPORAL_SCORER_HIDDEN_DIM="${TEMPORAL_SCORER_HIDDEN_DIM:-}"
TEMPORAL_SCORER_LAYERS="${TEMPORAL_SCORER_LAYERS:-}"
TEMPORAL_SCORER_DROPOUT="${TEMPORAL_SCORER_DROPOUT:-}"

# Distillation-specific inputs
IMAGE_FOLDER="${IMAGE_FOLDER:-${DATA_ROOT}/images}"
VIDEO_NAME="${VIDEO_NAME:-}"
VIDEO_NAMES="${VIDEO_NAMES:-}"
ALL_VIDEOS="${ALL_VIDEOS:-false}"
DETECTION_CONF="${DETECTION_CONF:-0.3}"
DETECTION_CACHE="${DETECTION_CACHE:-}"

case "$TRAIN_MODE" in
  distill|test) ;;
  *)
    echo "Unsupported TRAIN_MODE='${TRAIN_MODE}'. Use 'distill' or 'test'." >&2
    exit 1
    ;;
esac

if [[ ! -f "${APPTAINER_IMAGE}" ]]; then
  echo "Apptainer image not found: ${APPTAINER_IMAGE}" >&2
  exit 1
fi

if [[ ! -f "${CFG_PATH}" ]]; then
  echo "Model config not found: ${CFG_PATH}" >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

if [[ ! -f "${DETECTOR_PATH}" ]]; then
  echo "Detector weights not found: ${DETECTOR_PATH}" >&2
  exit 1
fi

if [[ ! -f "${COMMON_PY}" ]]; then
  echo "Frame extraction helper not found: ${COMMON_PY}" >&2
  exit 1
fi

case "$TRAIN_MODE" in
  distill)
    if ! is_true "$ALL_VIDEOS" && [[ -z "$VIDEO_NAME" && -z "$VIDEO_NAMES" ]]; then
      VIDEO_NAME="clip_2"
    fi
    : "${MAX_STEPS:=10000}"
    : "${LOG_EVERY:=25}"
    : "${SAVE_EVERY:=250}"
    : "${SAMPLE_LIMIT:=0}"
    ;;
  test)
    if is_true "$ALL_VIDEOS"; then
      echo "TRAIN_MODE=test uses a single video; ignoring ALL_VIDEOS=${ALL_VIDEOS}."
    fi
    ALL_VIDEOS=false
    VIDEO_NAME="$(pick_random_eligible_distill_video "$IMAGE_FOLDER" "$POSE_DIR" "$INTRINSICS_DIR" "$VIDEO_NAME" "$VIDEO_NAMES")"
    : "${MAX_STEPS:=100}"
    : "${LOG_EVERY:=10}"
    : "${SAVE_EVERY:=50}"
    : "${SAMPLE_LIMIT:=64}"
    echo "Test mode selected video: ${VIDEO_NAME}"
    ;;
esac

declare -a SELECTED_VIDEOS=()
if is_true "$ALL_VIDEOS"; then
  while IFS= read -r selected_video; do
    [[ -n "$selected_video" ]] || continue
    SELECTED_VIDEOS+=("$selected_video")
  done < <(list_candidate_videos "$IMAGE_FOLDER")
elif [[ -n "${VIDEO_NAMES}" ]]; then
  IFS='|' read -r -a SELECTED_VIDEOS <<< "${VIDEO_NAMES}"
else
  SELECTED_VIDEOS=("${VIDEO_NAME}")
fi

if (( ${#SELECTED_VIDEOS[@]} == 0 )); then
  echo "No candidate videos were resolved under ${IMAGE_FOLDER} for training." >&2
  exit 1
fi

SOURCE_IMAGE_FOLDER="${IMAGE_FOLDER}"
prepare_image_root_for_finetune "${SOURCE_IMAGE_FOLDER}" "${SELECTED_VIDEOS[@]}"
IMAGE_FOLDER="${TEMP_IMAGE_ROOT}"

if [[ -z "${RUN_NAME:-}" ]]; then
  if [[ -n "${EXPERIMENT_NAME}" ]]; then
    RUN_NAME="${EXPERIMENT_NAME}"
  elif [[ "$TRAIN_MODE" == "distill" ]]; then
    if is_true "$ALL_VIDEOS"; then
      RUN_NAME="distill_all_videos"
    elif [[ -n "${VIDEO_NAMES}" ]]; then
      RUN_NAME="distill_multi_video"
    else
      RUN_NAME="distill_${VIDEO_NAME}"
    fi
  elif [[ "$TRAIN_MODE" == "test" ]]; then
    RUN_NAME="test_${VIDEO_NAME}"
  fi
fi
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"

mkdir -p "${OUTPUT_ROOT}" "${RUN_OUTPUT_DIR}"

# ================ COMMAND BUILD ================

if [[ -z "$DETECTION_CACHE" ]]; then
  if is_true "$ALL_VIDEOS"; then
    DETECTION_CACHE="${RUN_OUTPUT_DIR}/detections_all_videos.json"
  elif [[ -n "$VIDEO_NAMES" ]]; then
    DETECTION_CACHE="${RUN_OUTPUT_DIR}/detections_selected_videos.json"
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
  --validation_split "${VALIDATION_SPLIT}"
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

append_value_flag_if_set "--loss_config" "${LOSS_CONFIG}"
append_value_flag_if_set "--experiment_name" "${EXPERIMENT_NAME}"

if [[ "$SAMPLE_LIMIT" != "0" ]]; then
  PYTHON_CMD+=(--sample_limit "${SAMPLE_LIMIT}")
fi

PYTHON_CMD+=(--detection_cache "${DETECTION_CACHE}")

if is_true "$ALL_VIDEOS"; then
  PYTHON_CMD+=(--all_videos)
elif [[ -n "${VIDEO_NAMES}" ]]; then
  IFS='|' read -r -a SELECTED_VIDEOS <<< "${VIDEO_NAMES}"
  for selected_video in "${SELECTED_VIDEOS[@]}"; do
    [[ -n "${selected_video}" ]] || continue
    PYTHON_CMD+=(--video "${selected_video}")
  done
else
  PYTHON_CMD+=(--video "${VIDEO_NAME}")
fi

append_value_flag_if_set "--temporal_window_size" "${TEMPORAL_WINDOW_SIZE}"
append_value_flag_if_set "--temporal_window_stride" "${TEMPORAL_WINDOW_STRIDE}"
append_value_flag_if_set "--temporal_max_frame_gap" "${TEMPORAL_MAX_FRAME_GAP}"
append_value_flag_if_set "--temporal_reduction" "${TEMPORAL_REDUCTION}"
append_value_flag_if_set "--temporal_scorer_hidden_dim" "${TEMPORAL_SCORER_HIDDEN_DIM}"
append_value_flag_if_set "--temporal_scorer_layers" "${TEMPORAL_SCORER_LAYERS}"
append_value_flag_if_set "--temporal_scorer_dropout" "${TEMPORAL_SCORER_DROPOUT}"
append_value_flag_if_set "--vipe_camera_weight" "${VIPE_CAMERA_WEIGHT}"
append_optional_bool_override "vipe_camera_enabled" "${VIPE_CAMERA_ENABLED}"
append_optional_bool_override "temporal_camera_enabled" "${TEMPORAL_CAMERA_ENABLED}"
append_value_flag_if_set "--temporal_camera_formulation" "${TEMPORAL_CAMERA_FORMULATION}"
append_value_flag_if_set "--temporal_camera_weight" "${TEMPORAL_CAMERA_WEIGHT}"
append_value_flag_if_set "--temporal_camera_scorer_weight" "${TEMPORAL_CAMERA_SCORER_WEIGHT}"
append_optional_bool_override "temporal_bbox_projected_enabled" "${TEMPORAL_BBOX_PROJECTED_ENABLED}"
append_value_flag_if_set "--temporal_bbox_projected_formulation" "${TEMPORAL_BBOX_PROJECTED_FORMULATION}"
append_value_flag_if_set "--temporal_bbox_projected_weight" "${TEMPORAL_BBOX_PROJECTED_WEIGHT}"
append_value_flag_if_set "--temporal_bbox_projected_scorer_weight" "${TEMPORAL_BBOX_PROJECTED_SCORER_WEIGHT}"
append_optional_bool_override "temporal_bbox_input_enabled" "${TEMPORAL_BBOX_INPUT_ENABLED}"
append_value_flag_if_set "--temporal_bbox_input_formulation" "${TEMPORAL_BBOX_INPUT_FORMULATION}"
append_value_flag_if_set "--temporal_bbox_input_weight" "${TEMPORAL_BBOX_INPUT_WEIGHT}"
append_value_flag_if_set "--temporal_bbox_input_scorer_weight" "${TEMPORAL_BBOX_INPUT_SCORER_WEIGHT}"

append_bool_flag "use_gpu" "${USE_GPU}"
append_bool_flag "amp" "${AMP}"

echo "Project root:    ${PROJECT_ROOT}"
echo "Model root:      ${MODEL_ROOT}"
echo "Output dir:      ${RUN_OUTPUT_DIR}"
echo "Train mode:      ${TRAIN_MODE}"
echo "Loss config:     ${LOSS_CONFIG:-<none>}"
echo "Experiment:      ${EXPERIMENT_NAME:-<none>}"
echo "Train scope:     ${TRAIN_SCOPE}"
echo "Max steps:       ${MAX_STEPS}"
echo "Log every:       ${LOG_EVERY}"
echo "Save every:      ${SAVE_EVERY}"
echo "Sample limit:    ${SAMPLE_LIMIT}"
echo "Validation split:${VALIDATION_SPLIT}"
echo "Checkpoint:      ${CHECKPOINT}"
echo "Source images:   ${SOURCE_IMAGE_FOLDER}"
echo "Prepared images: ${IMAGE_FOLDER}"
if is_true "$ALL_VIDEOS"; then
  echo "Videos:          all discovered videos"
elif [[ -n "${VIDEO_NAMES}" ]]; then
  echo "Videos:          ${VIDEO_NAMES}"
else
  echo "Video:           ${VIDEO_NAME}"
fi
if [[ -n "${TEMPORAL_WINDOW_SIZE}" || -n "${TEMPORAL_WINDOW_STRIDE}" || -n "${TEMPORAL_MAX_FRAME_GAP}" ]]; then
  echo "Temporal win:    size=${TEMPORAL_WINDOW_SIZE:-<default>} stride=${TEMPORAL_WINDOW_STRIDE:-<default>} gap=${TEMPORAL_MAX_FRAME_GAP:-<default>}"
fi
echo "Pose dir:        ${POSE_DIR}"
echo "Intrinsics dir:  ${INTRINSICS_DIR}"
print_command "${PYTHON_CMD[@]}"
echo "[progress] About to launch Apptainer via srun."
echo "[progress] SLURM wrapper log: ${outfile}"
echo "[progress] Python run output dir: ${RUN_OUTPUT_DIR}"

# ================ EXECUTION ================

APPTAINER_ARGS=(
  exec
  --nv
  --bind /scratch:/scratch
  --bind "${TEMP_PARENT}:${TEMP_PARENT}"
)

echo "[progress] Starting containerized training process now..."
srun apptainer "${APPTAINER_ARGS[@]}" "${APPTAINER_IMAGE}" "${PYTHON_CMD[@]}"
echo "[progress] Containerized training process exited cleanly."

echo "==============================================="
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
hours=$(printf "%.2f" "$(echo "$elapsed/3600" | bc -l)")

echo "Job finished at: $(date)"
echo "Execution took $hours hours"
echo "Writing output to $outfile"
echo "==============================================="
