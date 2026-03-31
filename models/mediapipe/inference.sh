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
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=%x.out

# ================ OUTPUT FILES ================
base_name="${SLURM_JOB_NAME}"
dir="SLURM_logs"
mkdir -p "$dir"
count=$(printf "%03d" $(($(ls "$dir" 2>/dev/null | grep -c "^${base_name}_[0-9]\\+\\.out$") + 1)))
outfile="${dir}/${base_name}_${count}.out"

exec >"$outfile" 2>&1

# ================ SLURM SETUP ================
module load 2024r1

# ================ CODE EXECUTION ================
echo "Loaded modules:"
module list 2>&1

echo "================ SLURM JOB INFO ================"
echo "Job ID:         $SLURM_JOB_ID"
echo "Job Name:       $SLURM_JOB_NAME"
echo "Partition:      $SLURM_JOB_PARTITION"
echo "Node List:      $SLURM_JOB_NODELIST"
echo "CPUs per task:  $SLURM_CPUS_PER_TASK"
echo "Memory per CPU: $SLURM_MEM_PER_CPU"
echo "Submit dir:     $SLURM_SUBMIT_DIR"
echo "Work dir:       $(pwd)"
echo "Job started at: $(date)"
start_time=$(date +%s)
echo "==============================================="

VIDEO_DIR="${VIDEO_DIR:-/scratch/mthakur/manifold/data/images}"
VIDEO_NAME="${VIDEO_NAME:-}"
VIDEO_FILE="${VIDEO_FILE:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/mthakur/manifold/outputs/mediapipe}"
TARGET_FPS="${TARGET_FPS:-30}"
VISUALIZE="${VISUALIZE:-true}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-/scratch/mthakur/manifold/models/mediapipe/apptainer/template.sif}"

if [[ -z "${VIDEO_NAME}" && -z "${VIDEO_FILE}" ]]; then
    echo "Set VIDEO_NAME or VIDEO_FILE before submitting this script." >&2
    exit 1
fi

visualize_flag="--visualize"
if [[ ! "${VISUALIZE}" =~ ^([Tt][Rr][Uu][Ee]|[Yy][Ee]?[Ss]|[Oo][Nn]|1)$ ]]; then
    visualize_flag="--no-visualize"
fi

cmd=(
  python main.py
  --video_folder "${VIDEO_DIR}"
  --output_folder "${OUTPUT_ROOT}"
  --target_fps "${TARGET_FPS}"
  "${visualize_flag}"
)

if [[ -n "${VIDEO_NAME}" ]]; then
    cmd+=(--video "${VIDEO_NAME}")
fi

if [[ -n "${VIDEO_FILE}" ]]; then
    cmd+=(--video_file "${VIDEO_FILE}")
fi

srun apptainer exec \
  --bind /scratch:/scratch \
  "${APPTAINER_IMAGE}" \
  bash -lc "cd /scratch/mthakur/manifold/models/mediapipe && $(printf '%q ' "${cmd[@]}")"

echo "==============================================="
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
hours=$(printf "%.2f" "$(echo "$elapsed/3600" | bc -l)")

echo "Job finished at: $(date)"
echo "Execution took $hours hours"
echo "Writing output to $outfile"
echo "==============================================="
