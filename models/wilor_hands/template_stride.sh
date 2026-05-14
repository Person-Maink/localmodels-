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
#SBATCH --partition=gpu-a100
#SBATCH --time=__TIME__
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=24G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=%x.out

SCRIPT_DIR="/scratch/mthakur/manifold/models/wilor_hands"
export MODE="stride"
export VIDEO_DIR="__VIDEO_DIR__"
export VIDEO_NAME="__NAME__"
export VIDEO_FILE="__FILE__"
export OUTPUT_ROOT="__OUTPUT_ROOT__"
export WILOR_CACHE_ROOT="__WILOR_CACHE_ROOT__"
export FRAME_CACHE_ROOT="__FRAME_CACHE_ROOT__"
export STRIDE_CONFIG_PATH="__STRIDE_CONFIG_PATH__"
export STRIDE_BACKEND="__STRIDE_BACKEND__"
export APPTAINER_IMAGE="__APPTAINER_IMAGE__"

bash "${SCRIPT_DIR}/inference.sh"
