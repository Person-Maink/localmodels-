# apptainer exec \
#   --nv \
#   --bind ~/.cache/huggingface:/home/mthakur/.cache/huggingface \
#   --bind ~/.cache/torch:/home/mthakur/.cache/torch \
#   /scratch/mthakur/vipe/apptainer/template.sif \
#   bash -c '
#     cd /scratch/mthakur/vipe

#     source /opt/conda/etc/profile.d/conda.sh
#     conda activate vipe

#     python - << "EOF"
# import torch
# import numpy as np

# from vipe.priors.depth.unidepth.models.unidepthv2.unidepthv2 import UniDepthV2, Pinhole

# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Load config exactly as UniDepth expects

# # Instantiate model CORRECTLY
# model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
# model.to(device).eval()

# print("Model loaded OK")

# # Dummy RGB input (C, H, W) float32
# rgb = torch.rand(3, 384, 512, device=device)

# # Dummy pinhole camera
# K = torch.tensor(
#     [[500.0, 0.0, 256.0],
#      [0.0, 500.0, 192.0],
#      [0.0, 0.0, 1.0]],
#     device=device
# )
# camera = Pinhole(K=K)

# with torch.no_grad():
#     preds = model.infer(rgb, camera)

# print("Prediction keys:", preds.keys())
# print("Depth shape:", preds["depth"].shape)
# print("Confidence shape:", preds["confidence"].shape)

# print("SUCCESS: ViPE vendored UniDepthV2 works when instantiated correctly.")
# EOF
#   '


# apptainer exec \
#   --bind ~/.cache/torch:/home/mthakur/.cache/torch \
#   /scratch/mthakur/vipe/apptainer/template.sif \
#   bash -c '
#     source /opt/conda/etc/profile.d/conda.sh
#     conda activate vipe

#     python - << "EOF"
# import torch
# from vipe.priors.depth.videodepthanything import VideoDepthAnythingDepthModel

# # Example to trigger model download
# model = VideoDepthAnythingDepthModel(model="vits")
# print("Model downloaded successfully.")
# EOF
#   '

apptainer exec \
  --bind ~/.cache/torch:/home/mthakur/.cache/torch \
  --bind ~/.cache/huggingface:/home/mthakur/.cache/huggingface \
  /scratch/mthakur/vipe/apptainer/template.sif \
  bash -c '
    source /opt/conda/etc/profile.d/conda.sh
    conda activate vipe

    python - << "EOF"
from huggingface_hub import snapshot_download

# Download PriorDepthAnything model
model_name = "Rain729/Prior-Depth-Anything"
snapshot_download(
    repo_id=model_name,
    local_files_only=False
)
EOF
  '
