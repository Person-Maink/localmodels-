import math
import sys

import cv2


video = sys.argv[1]

cap = cv2.VideoCapture(video)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {video}")

frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
cap.release()

# Mediapipe is CPU-only here and relatively light, but give some buffer for
# visualization and container startup.
sec_per_frame_at_1mp = 0.18
pixel_scale = (width * height) / 1_000_000

est_seconds = frames * sec_per_frame_at_1mp * max(pixel_scale, 0.25)
est_seconds *= 1.2
est_seconds += 45
est_seconds = int(math.ceil(est_seconds))

h = est_seconds // 3600
m = (est_seconds % 3600) // 60
s = est_seconds % 60

print(f"{h:02d}:{m:02d}:{s:02d}")
