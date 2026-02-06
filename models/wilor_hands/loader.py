import cv2
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from wilor.datasets.vitdet_dataset import ViTDetDataset

def load_images_from_folder(folder, file_types=("*.jpg", "*.png", "*.jpeg"), video_exts=(".mp4", ".mkg", ".mts", ".avi")):
    folder = Path(folder)
    paths = []

    for ext in file_types:
        paths.extend(folder.glob(ext))


    for video_file in folder.iterdir():
        if video_file.suffix.lower() in video_exts:
            video_name = video_file.stem

            out_dir = folder / f"{video_name}_frames"
            out_dir.mkdir(exist_ok=True)

            if out_dir.exists():
                frame_paths = sorted(out_dir.glob("*.jpg"))
                if frame_paths:
                    paths.extend(frame_paths)
                    print(f"Found existing frames for {video_name}, skipping extraction.")
                    continue

            cap = cv2.VideoCapture(str(video_file))
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_path = out_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                paths.append(frame_path)
                frame_idx += 1

            cap.release()

    return paths


def make_dataloader(model_cfg, img_cv2, boxes, right, rescale_factor=2.0, batch_size=16):
    dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=rescale_factor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
