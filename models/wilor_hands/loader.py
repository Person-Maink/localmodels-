import cv2
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader

from frame_store import FrameStore
from wilor.datasets.vitdet_dataset import ViTDetDataset

SUPPORTED_VIDEO_EXTS = (".mp4", ".avi", ".mts", ".mov")


def load_images_from_folder(
    folder,
    file_types=("*.jpg", "*.png", "*.jpeg"),
    video_exts=SUPPORTED_VIDEO_EXTS,
    frame_cache_root=None,
):
    folder = Path(folder)
    paths = []

    for ext in file_types:
        paths.extend(sorted(folder.glob(ext)))

    frame_store = FrameStore(folder, cache_root=frame_cache_root)
    for video_name in frame_store.list_videos():
        if video_name == "single_images":
            continue
        for frame_record in frame_store.iter_video_frames(video_name):
            if frame_record.source_kind in {"loose_frames", "single_image"}:
                paths.append(Path(frame_record.source_key))
            else:
                paths.append(Path(f"{video_name}/{frame_record.frame_name}.jpg"))

    return sorted(paths)


def make_dataloader(model_cfg, img_cv2, boxes, right, rescale_factor=2.0, batch_size=16):
    dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=rescale_factor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
