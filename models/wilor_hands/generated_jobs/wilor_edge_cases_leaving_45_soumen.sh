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
#SBATCH --time=01:33:36
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=10G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=%x.out

SCRIPT_DIR="/scratch/mthakur/manifold/models/wilor_hands"
export MODE="wilor"
export VIDEO_DIR="/scratch/mthakur/manifold/data/test"
export VIDEO_NAME="edge_cases_leaving_45_soumen"
export VIDEO_FILE="edge_cases_leaving_45_soumen.mp4"
export OUTPUT_ROOT="/scratch/mthakur/manifold/outputs/wilor"
export FRAME_CACHE_ROOT="/scratch/mthakur/manifold/data/test"
export APPTAINER_IMAGE="/scratch/mthakur/manifold/models/wilor_hands/apptainer/template.sif"

bash "${SCRIPT_DIR}/inference.sh"
