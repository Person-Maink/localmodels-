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
#SBATCH --time=09:47:18
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=64G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=%x.out

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
module load cuda/12.9

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

apptainer exec --nv \
    --bind /scratch:/scratch \
    --bind ~/.cache/torch:/home/mthakur/.cache/torch \
    --bind ~/.cache/huggingface:/home/mthakur/.cache/huggingface \
    /scratch/mthakur/manifold/models/vipe/apptainer/template.sif \
    bash -c 'cd /scratch/mthakur/manifold/models/vipe && /opt/conda/bin/conda run -n vipe python run.py pipeline=no_vda streams=raw_mp4_stream streams.base_path="/scratch/mthakur/manifold/data/images/Ipe9xJCfuTM.mp4" streams.frame_end=-1 pipeline.output.path=/scratch/mthakur/manifold/outputs/vipe/ pipeline.output.save_artifacts=true pipeline.output.save_viz=false'

# apptainer exec --nv \
#     --bind /scratch/mthakur/manifold/data/:/data/ \
#     --bind /scratch/mthakur/manifold/outputs/vipe/:/output/ \
#     --bind ~/.cache/torch:/home/mthakur/.cache/torch \
#     --bind ~/.cache/huggingface:/home/mthakur/.cache/huggingface \
#     /scratch/mthakur/manifold/models/vipe/apptainer/template.sif \
#     bash -c '/opt/conda/bin/conda run -n vipe vipe infer "data/Ipe9xJCfuTM.mp4" --output /output/ --pipeline no_vda'

echo "==============================================="
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
hours=$(printf "%.2f" "$(echo "$elapsed/3600" | bc -l)")

echo "Job finished at: $(date)"
echo "Execution took $hours hours"
echo "Writing output to $outfile"
echo "==============================================="
