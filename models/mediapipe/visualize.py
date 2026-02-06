import cv2
import pandas as pd
import numpy as np
import os

# 21 landmark connections for Mediapipe Hands
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

def overlay_hand_pose(video_path, csv_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f"[SKIP] Inference was empty for {csv_path}")
        return

    if df.empty:
        print(f"[SKIP] Inference was empty for {csv_path}")
        return



    df = pd.read_csv(csv_path)

    # Expect columns: frame_id, joint_id, x, y, z, visibility (if present)
    grouped = df.groupby("frame_id")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in grouped.groups:
            frame_data = grouped.get_group(frame_idx)

            # draw joints
            for _, row in frame_data.iterrows():
                if not np.isnan(row["x"]):
                    x = int(row["x"] * width)
                    y = int(row["y"] * height)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # draw bone connections
            for (i, j) in HAND_CONNECTIONS:
                try:
                    p1 = frame_data.loc[frame_data["joint_id"] == i].iloc[0]
                    p2 = frame_data.loc[frame_data["joint_id"] == j].iloc[0]
                    if not (np.isnan(p1["x"]) or np.isnan(p2["x"])):
                        x1, y1 = int(p1["x"] * width), int(p1["y"] * height)
                        x2, y2 = int(p2["x"] * width), int(p2["y"] * height)
                        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                except IndexError:
                    continue

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Overlay video saved to: {output_path}")
