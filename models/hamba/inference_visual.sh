module load 2025 visual virtualgl slurm
cd /scratch/mthakur/Hamba/
vglrun apptainer exec   --nv   /scratch/mthakur/Hamba/apptainer/template.sif   bash -c "python demo.py --checkpoint ckpts/hamba/checkpoints/hamba.ckpt --img_folder example_data --out_folder ./demo_out/example_data/ --full_frame"
