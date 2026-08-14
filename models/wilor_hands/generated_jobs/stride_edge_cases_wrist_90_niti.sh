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

#SBATCH --job-name=stride-inference
#SBATCH --partition=gpu-a100-small
#SBATCH --time=01:37:12
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=24G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=%x.out

SCRIPT_DIR="/scratch/mthakur/manifold/models/wilor_hands"
export MODE="stride"
export VIDEO_DIR="/scratch/mthakur/manifold/data/test"
export VIDEO_NAME="edge_cases_wrist_90_niti"
export VIDEO_FILE="edge_cases_wrist_90_niti.mp4"
export OUTPUT_ROOT="/scratch/mthakur/manifold/outputs/stride"
export WILOR_CACHE_ROOT="/scratch/mthakur/manifold/outputs/wilor"
export FRAME_CACHE_ROOT="/scratch/mthakur/manifold/data/test"
export STRIDE_CONFIG_PATH="/scratch/mthakur/manifold/models/wilor_hands/stride_configs/hmp.yaml"
export STRIDE_BACKEND="hmp"
export APPTAINER_IMAGE="/scratch/mthakur/manifold/models/wilor_hands/apptainer/template-hmp.sif"

bash "${SCRIPT_DIR}/inference.sh"
