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
#SBATCH --time=01:24:37
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=24G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=%x.out

# ================ ENV VARIABLES ================

export HYDRA_FULL_ERROR=1
export LD_LIBRARY_PATH=/cm/local/apps/gcc/10.2.0/lib64/libstdc++.so.6.0.28:$LD_LIBRARY_PATH

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
module load python/3.10.13

# ================ CODE EXECUTION ================

echo "Loaded modules:"
module list 2>&1

nvidia-smi

echo "================ SLURM JOB INFO ================"
echo "Job ID:         $SLURM_JOB_ID"
echo "Job Name:       $SLURM_JOB_NAME"
echo "Partition:      $SLURM_JOB_PARTITION"
echo "Node List:      $SLURM_JOB_NODELIST"
echo "CPUs per task:  $SLURM_CPUS_PER_TASK"
echo "Memory per CPU: $SLURM_MEM_PER_CPU"
echo "GPUs per task:  $SLURM_GPUS_PER_TASK"
echo "Memory per GPU: $SLURM_MEM_PER_GPU"
echo "Submit dir:     $SLURM_SUBMIT_DIR"
echo "Work dir:       $(pwd)"
echo "Job started at: $(date)"
start_time=$(date +%s)
echo "==============================================="

PROJECT_ROOT="/scratch/mthakur/manifold"
MODEL_ROOT="${PROJECT_ROOT}/models/dyn-hamr"
DATA_ROOT="${PROJECT_ROOT}/data"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/dynhamr"
LOG_ROOT="${OUTPUT_ROOT}/logs"
VIDEO_DIR="images"
VIDEO_NAME="me 1"
VIDEO_EXT="mp4"
IS_STATIC="${IS_STATIC:-False}"
RUN_PRIOR="${RUN_PRIOR:-False}"
RUN_VIS="${RUN_VIS:-True}"
TEMPORAL_SMOOTH="${TEMPORAL_SMOOTH:-False}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:--1}"
ROOT_ITERS="${ROOT_ITERS:-40}"
SMOOTH_ITERS="${SMOOTH_ITERS:-60}"
DETECTRON2_CKPT="${DETECTRON2_CKPT:-/home/mthakur/.cache/torch/hub/detectron2/model_final_f05665.pkl}"

export APPTAINER_IMAGE="${MODEL_ROOT}/apptainer/template.sif"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

VIDEO_PATH="${DATA_ROOT}/${VIDEO_DIR}/${VIDEO_NAME}.${VIDEO_EXT}"
HMP_FRAME_DIR="${DATA_ROOT}/images/${VIDEO_NAME}"
if [[ ! -f "${VIDEO_PATH}" ]]; then
  echo "Video not found: ${VIDEO_PATH}" >&2
  exit 1
fi

echo "Using video: ${VIDEO_PATH}"
if [[ -f "${DETECTRON2_CKPT}" ]]; then
  echo "Using local Detectron2 checkpoint: ${DETECTRON2_CKPT}"
  export APPTAINERENV_HAMER_DETECTRON2_CKPT="${DETECTRON2_CKPT}"
else
  echo "Local Detectron2 checkpoint not found at ${DETECTRON2_CKPT}; HaMeR will try downloading." >&2
  unset APPTAINERENV_HAMER_DETECTRON2_CKPT
fi

srun apptainer exec \
  --nv \
  --bind ~/.cache/torch:/home/mthakur/.cache/torch \
  --bind ~/.cache/huggingface:/home/mthakur/.cache/huggingface \
  --bind /scratch:/scratch \
  "${APPTAINER_IMAGE}" \
  python -u "${MODEL_ROOT}/dyn-hamr/run_opt.py" \
  data=video \
  run_opt=True \
  "run_vis=${RUN_VIS}" \
  "run_prior=${RUN_PRIOR}" \
  "data.root=${DATA_ROOT}" \
  "data.video_dir=${VIDEO_DIR}" \
  "data.seq=${VIDEO_NAME}" \
  "data.ext=${VIDEO_EXT}" \
  "data.src_path=${VIDEO_PATH}" \
  "data.start_idx=${START_IDX}" \
  "data.end_idx=${END_IDX}" \
  "is_static=${IS_STATIC}" \
  "temporal_smooth=${TEMPORAL_SMOOTH}" \
  "HMP.vid_path=${HMP_FRAME_DIR}" \
  "optim.root.num_iters=${ROOT_ITERS}" \
  "optim.smooth.num_iters=${SMOOTH_ITERS}" \
  "log_root=${LOG_ROOT}"

echo "==============================================="
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
hours=$(printf "%.2f" "$(echo "$elapsed/3600" | bc -l)")

echo "Job finished at: $(date)"
echo "Execution took $hours hours"
echo "Writing output to $outfile"
echo "==============================================="
