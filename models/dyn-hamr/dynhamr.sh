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

#SBATCH --job-name=example
#SBATCH --partition=gpu-a100-small
#SBATCH --time=00:25:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=10G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=%x.out

# ================ ENV VARIABLES ================

export HYDRA_FULL_ERROR=1
export LD_LIBRARY_PATH=/cm/local/apps/gcc/10.2.0/lib64/libstdc++.so.6.0.28:$LD_LIBRARY_PATH # add to ~/zshrc OR bashrc whatever

# ================ OUTPUT FILES ================

# compute a small incremental index based on existing files
base_name="${SLURM_JOB_NAME}"
dir="SLURM_logs"
mkdir -p "$dir"
count=$(printf "%03d" $(($(ls "$dir" 2>/dev/null | grep -c "^${base_name}_[0-9]\+\.out$") + 1)))
outfile="${dir}/${base_name}_${count}.out"

# redirect stdout and stderr
exec >"$outfile" 2>&1

# ================ SLURM SETUP ================

# Load modules:
module load 2024r1
# module load miniconda3
module load cuda/11.7
# module load openmpi/4.1.6
module load python/3.10.13
# module load py-torch/1.12.1
# module load gcc/11.3.0
# module load py-pip
# module load py-numpy
# module load py-pyyaml
# module load py-tqdm
# module load ffmpeg

# ================ CODE EXECUTION ================

cd /scratch/mthakur/Dyn-HaMR
source .dynhamr/bin/activate

echo "Activated environment: $VIRTUAL_ENV"
echo "Python version:"
.dynhamr/bin/python3.10 --version
echo "Loaded modules:"
module list 2>&1

# echo "Loaded Python Libraries"
# .dynhamr/bin/pip freeze 2>&1

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

# .dynhamr/bin/python3 -u dyn-hamr/run_opt.py data=video run_opt=True run_vis=False is_static=True
.dynhamr/bin/python3.10 -u dyn-hamr/run_opt.py data=video run_opt=True data.seq=demo1 is_static=True

echo "==============================================="
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
hours=$(printf "%.2f" "$(echo "$elapsed/3600" | bc -l)")

echo "Job finished at: $(date)"
echo "Execution took $hours hours"
echo "Writing output to $outfile"
echo "==============================================="
