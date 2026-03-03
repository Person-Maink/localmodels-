import cv2
import sys
import math

video = sys.argv[1]

cap = cv2.VideoCapture(video)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {video}")

fps = cap.get(cv2.CAP_PROP_FPS)
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
cap.release()

# --- model parameters (tune once if needed) ---
SEC_PER_FRAME_AT_1MP = 4.37  # empirical, conservative
PIXELS = width * height
PIX_SCALE = PIXELS / 1_000_000

est_seconds = frames * SEC_PER_FRAME_AT_1MP * PIX_SCALE
est_seconds *= 1.2  # safety margin
est_seconds += 15  # startup time and all

est_seconds = int(math.ceil(est_seconds))

h = est_seconds // 3600
m = (est_seconds % 3600) // 60
s = est_seconds % 60

print(f"{h:02d}:{m:02d}:{s:02d}")
