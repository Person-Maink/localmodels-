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

#SBATCH --job-name=vipe-inference
#SBATCH --partition=gpu-a100
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=64G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --array=0-2%8
#SBATCH --output=%x_%A_%a.bootstrap.out

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MANIFEST="${SCRIPT_DIR}/../manifests/bucket_02-00-00_20260326_112355.tsv"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  exit 1
fi

manifest_line=$(sed -n "$((TASK_ID + 1))p" "${MANIFEST}")
if [[ -z "${manifest_line}" ]]; then
  echo "No manifest entry for task ${TASK_ID} in ${MANIFEST}" >&2
  exit 1
fi

IFS=$'\t' read -r SAFE_NAME VIDEO_NAME VIDEO_EXT VIDEO_FILE <<< "${manifest_line}"
if [[ -z "${SAFE_NAME}" || -z "${VIDEO_FILE}" ]]; then
  echo "Malformed manifest row for task ${TASK_ID}: ${manifest_line}" >&2
  exit 1
fi

# ================ OUTPUT FILES ================

base_name="${SLURM_JOB_NAME}"
dir="${SLURM_SUBMIT_DIR:-$(pwd)}/SLURM_logs"
mkdir -p "$dir"
job_token="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
outfile="${dir}/${base_name}_${job_token}_${TASK_ID}_${SAFE_NAME}.out"

exec >"$outfile" 2>&1

# ================ SLURM SETUP ================

module load 2024r1
module load cuda/12.9

# ================ CODE EXECUTION ================

echo "Loaded modules:"
module list 2>&1

nvidia-smi

echo "================ SLURM JOB INFO ================"
echo "Job ID:         $SLURM_JOB_ID"
echo "Array Job ID:   ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "Array Task ID:  ${TASK_ID}"
echo "Job Name:       $SLURM_JOB_NAME"
echo "Partition:      $SLURM_JOB_PARTITION"
echo "Node List:      $SLURM_JOB_NODELIST"
echo "CPUs per task:  $SLURM_CPUS_PER_TASK"
echo "Memory per CPU: $SLURM_MEM_PER_CPU"
echo "GPUs per task:  $SLURM_GPUS_PER_TASK"
echo "Memory per GPU: $SLURM_MEM_PER_GPU"
echo "Submit dir:     $SLURM_SUBMIT_DIR"
echo "Work dir:       $(pwd)"
echo "Manifest:       ${MANIFEST}"
echo "Video file:     ${VIDEO_FILE}"
echo "Job started at: $(date)"
start_time=$(date +%s)
echo "==============================================="

apptainer exec --nv \
    --bind /scratch:/scratch \
    --bind ~/.cache/torch:/home/mthakur/.cache/torch \
    --bind ~/.cache/huggingface:/home/mthakur/.cache/huggingface \
    /scratch/mthakur/manifold/models/vipe/apptainer/template.sif \
    bash -c "/opt/conda/bin/conda run -n vipe vipe infer \"/scratch/mthakur/manifold/data/images/${VIDEO_FILE}\" --output /scratch/mthakur/manifold/outputs/vipe/ --pipeline no_vda"

echo "==============================================="
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
hours=$(printf "%.2f" "$(echo "$elapsed/3600" | bc -l)")

echo "Job finished at: $(date)"
echo "Execution took $hours hours"
echo "Writing output to $outfile"
echo "==============================================="
