import cv2
import numpy as np

def iterate_video_frames(video_path, target_fps=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0
    def generator():
        nonlocal frame_idx
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if target_fps and fps > target_fps:
                if frame_idx % int(fps / target_fps) != 0:
                    frame_idx += 1
                    continue
            yield frame_idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_idx += 1
        cap.release()
    return generator(), fps

