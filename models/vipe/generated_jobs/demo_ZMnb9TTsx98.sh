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
#SBATCH --time=01:31:09
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

PROJECT_ROOT="/scratch/mthakur/manifold"
MODEL_ROOT="${PROJECT_ROOT}/models/vipe"
MODEL_ASSETS_ROOT="${MODEL_ASSETS_ROOT:-${PROJECT_ROOT}/models/model_assets}"
APPTAINER_IMAGE="${MODEL_ROOT}/apptainer/template.sif"
CHUNK_SECONDS="${CHUNK_SECONDS:-600}"
FRAME_SKIP="${FRAME_SKIP:-1}"
SAVE_VIZ="${SAVE_VIZ:-False}"

REQUIRED_ASSETS=(
  "${MODEL_ASSETS_ROOT}/vipe/droid_slam/droid.pth"
  "${MODEL_ASSETS_ROOT}/vipe/geocalib/pinhole.tar"
  "${MODEL_ASSETS_ROOT}/vipe/track_anything/sam_vit_b_01ec64.pth"
  "${MODEL_ASSETS_ROOT}/vipe/track_anything/R50_DeAOTL_PRE_YTB_DAV.pth"
  "${MODEL_ASSETS_ROOT}/vipe/track_anything/groundingdino_swint_ogc.pth"
  "${MODEL_ASSETS_ROOT}/vipe/huggingface/bert-base-uncased"
  "${MODEL_ASSETS_ROOT}/vipe/huggingface/unidepth-v2-vitl14"
  "${MODEL_ASSETS_ROOT}/vipe/priorda/depth_anything_v2_vitb.pth"
  "${MODEL_ASSETS_ROOT}/vipe/priorda/prior_depth_anything_vitb.pth"
)

for asset in "${REQUIRED_ASSETS[@]}"; do
  if [[ ! -e "${asset}" ]]; then
    echo "Required ViPE asset not found: ${asset}" >&2
    exit 1
  fi
done

export APPTAINERENV_MODEL_ASSETS_ROOT="${MODEL_ASSETS_ROOT}"
export APPTAINERENV_HF_HUB_OFFLINE=1
export APPTAINERENV_TRANSFORMERS_OFFLINE=1

apptainer exec --nv \
    --bind /scratch:/scratch \
    "${APPTAINER_IMAGE}" \
    bash -c "cd /scratch/mthakur/manifold/models/vipe && /opt/conda/bin/conda run -n vipe python run_chunked.py --video /scratch/mthakur/manifold/data/images/ZMnb9TTsx98.mp4 --output /scratch/mthakur/manifold/outputs/vipe/ --pipeline no_vda --chunk-seconds ${CHUNK_SECONDS} --frame-skip ${FRAME_SKIP} --skip-existing true --save-viz ${SAVE_VIZ}"

echo "==============================================="
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
hours=$(printf "%.2f" "$(echo "$elapsed/3600" | bc -l)")

echo "Job finished at: $(date)"
echo "Execution took $hours hours"
echo "Writing output to $outfile"
echo "==============================================="
