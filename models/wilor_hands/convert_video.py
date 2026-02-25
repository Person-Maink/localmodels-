import cv2
from pathlib import Path

RAW_DIR = Path("../../data/images")

VIDEO_EXTS = {".mp4", ".mts"}

for video in RAW_DIR.iterdir():
    if video.suffix.lower() not in VIDEO_EXTS:
        continue

    out_avi = video.with_suffix(".avi")
    if out_avi.exists():
        print(f"Skipping (exists): {out_avi.name}")
        continue

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(out_avi), fourcc, fps, (width, height))

    print(f"Converting: {video.name} -> {out_avi.name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    cap.release()
    writer.release()
