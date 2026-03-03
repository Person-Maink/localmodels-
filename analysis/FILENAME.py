from pathlib import Path

# Internal Path objects (single source of truth)
_OUTPUTS_ROOT = Path("/home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/outputs")
_ANALYSIS_ROOT = Path("/home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/analysis")

# Hamba root clip (comment in/out one line)
_HAMBA_CLIP = "163 (2) FU"
# _HAMBA_CLIP = "120-2_clip_1"  # show stable right hand
# _HAMBA_CLIP = "120-2_clip_1_amplified"  # show stable right hand
# _HAMBA_CLIP = "120-2_clip_2"
# _HAMBA_CLIP = "120-2_clip_2_amplified"
# _HAMBA_CLIP = "120-2_clip_3"
# _HAMBA_CLIP = "120-2_clip_3_amplified"
# _HAMBA_CLIP = "120-2_clip_4"  # show left hand
# _HAMBA_CLIP = "120-2_clip_4_amplified"  # show left hand
# _HAMBA_CLIP = "120-2_clip_5"  # both hands out
# _HAMBA_CLIP = "120-2_clip_5_amplified"  # both hands out
# _HAMBA_CLIP = "120-2_clip_6"  # not that good acc
# _HAMBA_CLIP = "120-2_clip_7"
# _HAMBA_CLIP = "120-2_clip_7_amplified"
# _HAMBA_CLIP = "124-7 DBS uit"
# _HAMBA_CLIP = "163 (2) FU"  # water pouring good angle
# _HAMBA_CLIP = "clip_1"  # hand shaking very fast
# _HAMBA_CLIP = "clip_1_amplified"  # hand shaking very fast
# _HAMBA_CLIP = "clip_2"
# _HAMBA_CLIP = "clip_2_amplified"
# _HAMBA_CLIP = "clip_3"
# _HAMBA_CLIP = "clip_3_amplified"
# _HAMBA_CLIP = "clip_4"

# WiLoR comparison clip (comment in/out one line)
_HAMBA_COMP_CLIP = "120-2_clip_1_amplified"  # show stable right hand
# _HAMBA_COMP_CLIP = "120-2_clip_2_amplified"
# _HAMBA_COMP_CLIP = "120-2_clip_3_amplified"
# _HAMBA_COMP_CLIP = "120-2_clip_4_amplified"
# _HAMBA_COMP_CLIP = "120-2_clip_5_amplified"
# _HAMBA_COMP_CLIP = "120-2_clip_7_amplified"
# _HAMBA_COMP_CLIP = "clip_1_amplified"
# _HAMBA_COMP_CLIP = "clip_2_amplified"
# _HAMBA_COMP_CLIP = "clip_3_amplified"

# WiLoR root clip (comment in/out one line)
# _WILOR_CLIP = "163 (2) FU"
_WILOR_CLIP = "120-2_clip_1"  # show stable right hand
# _WILOR_CLIP = "120-2_clip_1_amplified"  # show stable right hand
# _WILOR_CLIP = "120-2_clip_2"
# _WILOR_CLIP = "120-2_clip_2_amplified"
# _WILOR_CLIP = "120-2_clip_3"
# _WILOR_CLIP = "120-2_clip_3_amplified"
# _WILOR_CLIP = "120-2_clip_4"  # show left hand
# _WILOR_CLIP = "120-2_clip_4_amplified"  # show left hand
# _WILOR_CLIP = "120-2_clip_5"  # both hands out
# _WILOR_CLIP = "120-2_clip_5_amplified"  # both hands out
# _WILOR_CLIP = "120-2_clip_6"  # not that good acc
# _WILOR_CLIP = "120-2_clip_7"
# _WILOR_CLIP = "120-2_clip_7_amplified"
# _WILOR_CLIP = "124-7 DBS uit"
# _WILOR_CLIP = "163 (2) FU"  # water pouring good angle
# _WILOR_CLIP = "clip_1"  # hand shaking very fast
# _WILOR_CLIP = "clip_1_amplified"  # hand shaking very fast
# _WILOR_CLIP = "clip_2"
# _WILOR_CLIP = "clip_2_amplified"
# _WILOR_CLIP = "clip_3"
# _WILOR_CLIP = "clip_3_amplified"
# _WILOR_CLIP = "clip_4"

# WiLoR comparison clip (comment in/out one line)
_WILOR_COMP_CLIP = "120-2_clip_1_amplified"  # show stable right hand
# _WILOR_COMP_CLIP = "120-2_clip_2_amplified"
# _WILOR_COMP_CLIP = "120-2_clip_3_amplified"
# _WILOR_COMP_CLIP = "120-2_clip_4_amplified"
# _WILOR_COMP_CLIP = "120-2_clip_5_amplified"
# _WILOR_COMP_CLIP = "120-2_clip_7_amplified"
# _WILOR_COMP_CLIP = "clip_1_amplified"
# _WILOR_COMP_CLIP = "clip_2_amplified"
# _WILOR_COMP_CLIP = "clip_3_amplified"

# MediaPipe root file (comment in/out one line)
_MEDIAPIPE_CLIP = "120-2_clip_1_keypoints.csv"
# _MEDIAPIPE_CLIP = "120-2_clip_1_amplified_keypoints.csv"
# _MEDIAPIPE_CLIP = "120-2_clip_2_keypoints.csv"
# _MEDIAPIPE_CLIP = "120-2_clip_2_amplified_keypoints.csv"
# _MEDIAPIPE_CLIP = "120-2_clip_3_keypoints.csv"
# _MEDIAPIPE_CLIP = "120-2_clip_3_amplified_keypoints.csv"
# _MEDIAPIPE_CLIP = "120-2_clip_4_keypoints.csv"
# _MEDIAPIPE_CLIP = "120-2_clip_4_amplified_keypoints.csv"
# _MEDIAPIPE_CLIP = "120-2_clip_5_keypoints.csv"
# _MEDIAPIPE_CLIP = "120-2_clip_5_amplified_keypoints.csv"
# _MEDIAPIPE_CLIP = "120-2_clip_6_keypoints.csv"
# _MEDIAPIPE_CLIP = "120-2_clip_7_keypoints.csv"
# _MEDIAPIPE_CLIP = "120-2_clip_7_amplified_keypoints.csv"
# _MEDIAPIPE_CLIP = "124-6 DBS uit_keypoints.csv"
# _MEDIAPIPE_CLIP = "163 (2) FU_keypoints.csv"
# _MEDIAPIPE_CLIP = "clip_1_keypoints.csv"
# _MEDIAPIPE_CLIP = "clip_1_amplified_keypoints.csv"
# _MEDIAPIPE_CLIP = "clip_2_keypoints.csv"
# _MEDIAPIPE_CLIP = "clip_2_amplified_keypoints.csv"
# _MEDIAPIPE_CLIP = "clip_3_keypoints.csv"
# _MEDIAPIPE_CLIP = "clip_3_amplified_keypoints.csv"
# _MEDIAPIPE_CLIP = "clip_4_keypoints.csv"

# MediaPipe comparison file (comment in/out one line)
_MEDIAPIPE_COMP_CLIP = "120-2_clip_5_amplified_keypoints.csv"
# _MEDIAPIPE_COMP_CLIP = "120-2_clip_1_amplified_keypoints.csv"
# _MEDIAPIPE_COMP_CLIP = "120-2_clip_2_amplified_keypoints.csv"
# _MEDIAPIPE_COMP_CLIP = "120-2_clip_3_amplified_keypoints.csv"
# _MEDIAPIPE_COMP_CLIP = "120-2_clip_4_amplified_keypoints.csv"
# _MEDIAPIPE_COMP_CLIP = "120-2_clip_7_amplified_keypoints.csv"
# _MEDIAPIPE_COMP_CLIP = "clip_1_amplified_keypoints.csv"
# _MEDIAPIPE_COMP_CLIP = "clip_2_amplified_keypoints.csv"
# _MEDIAPIPE_COMP_CLIP = "clip_3_amplified_keypoints.csv"

# ViPE pose file
_VIPE_POSE_FILE = "120-2_clip_1.npz"

_MANO_RIGHT_PATH = _ANALYSIS_ROOT / "mano_data" / "MANO_RIGHT.pkl"
_WILOR_ROOT = _OUTPUTS_ROOT / "wilor" / _WILOR_CLIP / "meshes"
_WILOR_COMP = _OUTPUTS_ROOT / "wilor" / _WILOR_COMP_CLIP / "meshes"
_HAMBA_ROOT = _OUTPUTS_ROOT / "hamba" / _HAMBA_CLIP / "meshes"
_HAMBA_COMP = _OUTPUTS_ROOT / "hamba" / _HAMBA_COMP_CLIP / "meshes"
_MEDIAPIPE_ROOT = _OUTPUTS_ROOT / "mediapipe" / "keypoints" / _MEDIAPIPE_CLIP
_MEDIAPIPE_COMP = _OUTPUTS_ROOT / "mediapipe" / "keypoints" / _MEDIAPIPE_COMP_CLIP
_VIPE_ROOT = _OUTPUTS_ROOT / "vipe" / "pose"
_VIPE_POSE_FILE = _VIPE_ROOT / _VIPE_POSE_FILE

# Legacy alias with trailing slash expected by older scripts.
base = str(_OUTPUTS_ROOT) + "/"

# String exports for compatibility with existing consumers.
OUTPUTS_ROOT = str(_OUTPUTS_ROOT)
ANALYSIS_ROOT = str(_ANALYSIS_ROOT)
MANO_RIGHT_PATH = str(_MANO_RIGHT_PATH)
WILOR_ROOT = str(_WILOR_ROOT)
WILOR_COMP = str(_WILOR_COMP)
HAMBA_ROOT = str(_HAMBA_ROOT)
HAMBA_COMP = str(_HAMBA_COMP)
MEDIAPIPE_ROOT = str(_MEDIAPIPE_ROOT)
MEDIAPIPE_COMP = str(_MEDIAPIPE_COMP)
VIPE_ROOT = str(_VIPE_ROOT)
VIPE_POSE_FILE = str(_VIPE_POSE_FILE)
