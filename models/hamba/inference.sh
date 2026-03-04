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

#SBATCH --job-name=hamba-inference
#SBATCH --partition=gpu-a100
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=24G
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
# module load miniconda3
module load cuda/11.7
# module load openmpi/4.1.6
# module load python/3.10.13
# module load py-torch/1.12.1
# module load gcc/11.3.0
# module load py-pip
# module load py-numpy
# module load py-pyyaml
# module load py-tqdm
# module load ffmpeg

# ================ CODE EXECUTION ================

## Use this simple command to check that your sbatch
## settings are working (it should show the GPU that you requested)

# source /scratch/mthakur/Dyn-HaMR/.dynhamr/bin/activate
# echo "Activated environment: $VIRTUAL_ENV"
# echo "Python version:"
#  /scratch/mthakur/Dyn-HaMR/.dynhamr/bin/python3.10 --version
echo "Loaded modules:"
module list 2>&1

# echo "Loaded Python Libraries"
#  /scratch/mthakur/Dyn-HaMR/.dynhamr/bin/pip freeze 2>&1

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

srun apptainer exec \
  --nv \
  --bind /scratch:/scratch \
  /scratch/mthakur/manifold/models/hamba/apptainer/template.sif \
  python main.py
  # python demo.py --checkpoint ckpts/hamba/checkpoints/hamba.ckpt --img_folder example_data --out_folder ./demo_out/example_data/ --full_frame
  # python demo.py --checkpoint ckpts/hamba.ckpt --img_folder /scratch/mthakur/manifold/data/images/ --out_folder /demo_out/ --full_frame

# srun apptainer exec\
#   --nv\
#   --bind "/scratch/mthakur/DROID-SLAM/data/sfm_bench/rgb:/app/DROID-SLAM/data/sfm_bench/rgb"\
#   --bind "/scratch/mthakur/DROID-SLAM/calib/eth.txt:/app/DROID-SLAM/calib/eth.txt"\
#   "/scratch/mthakur/DROID-SLAM/apptainer/template.sif"\
#   python demo.py --imagedir=data/sfm_bench/rgb --calib=calib/eth.txt --save_path=saved_output --reconstruction_path my_reconstruction.pth
echo "==============================================="
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
hours=$(printf "%.2f" "$(echo "$elapsed/3600" | bc -l)")

echo "Job finished at: $(date)"
echo "Execution took $hours hours"
echo "Writing output to $outfile"
echo "==============================================="
