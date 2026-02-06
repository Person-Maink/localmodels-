#!/bin/bash

# Available HPC Partitions:
#   compute / compute-p1   : CPU jobs (48 CPUs, 185 GB RAM, Phase 1)
#   compute-p2             : CPU jobs (64 CPUs, 250 GB RAM, Phase 2)
#   gpu / gpu-v100         : GPU jobs (4x V100, 32 GB VRAM each, Phase 1)
#   gpu-a100               : GPU jobs (4x A100, 80 GB VRAM each, Phase 2)
#   gpu-a100-small         : Small GPU jobs (≤1 GPU, ≤10 GB VRAM, ≤2 CPUs, ≤4h)
#   memory                 : High-memory CPU jobs (>250 GB RAM)
#   visual                 : Visualization jobs

#SBATCH --job-name=mediapipe_inference
#SBATCH --partition=compute-p2
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --account=Education-EEMCS-MSc-DSAIT

module load 2023r1-gcc11
module load openmpi/4.1.4
module load python/3.10

echo "ERROR, ENTER THE CORRECT PATH"
cd $HOME/path/to/mediapipe/src

uv sync --frozen
uv run python main.py

# ---------------------------------------------------------------------
# Notes:
# - "uv run" executes Python inside the uv-managed venv.
# - You can pass args to main.py if needed:
#   uv run python main.py --video-folder "../../data/"
# ---------------------------------------------------------------------
