import os
import re
from pathlib import Path

import cv2


def _natural_key(name):
    base = os.path.splitext(os.path.basename(str(name)))[0]
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", base)]


def images_to_video(input_folder, output_path, fps=30):
    input_folder = Path(input_folder)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = [p for p in input_folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    images = sorted(images, key=_natural_key)
    if not images:
        raise ValueError(f"No images found in folder: {input_folder}")

    first_frame = cv2.imread(str(images[0]))
    if first_frame is None:
        raise ValueError(f"Failed to read first frame: {images[0]}")

    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for image_path in images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Warning: Skipping unreadable image {image_path.name}")
            continue
        writer.write(frame)

    writer.release()
