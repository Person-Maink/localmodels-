import argparse
import os

import cv2


def increase_contrast(input_path, output_path, clip_limit, tile_grid_size):
    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl, a_channel, b_channel))
        enhanced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        writer.write(enhanced_frame)

    capture.release()
    writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Increase the contrast of a video.")
    parser.add_argument("--input", type=str, default="me 1.mp4", help="Path to the input video")
    parser.add_argument("--output", type=str, default="me 1_contrast_150.mp4", help="Path to the output video")
    parser.add_argument(
        "--clip-limit",
        type=float,
        default=2.5,
        help="CLAHE clip limit. Higher values increase local contrast more aggressively.",
    )
    parser.add_argument(
        "--tile-grid-size",
        type=int,
        nargs=2,
        default=(8, 8),
        metavar=("W", "H"),
        help="CLAHE tile grid size, passed as two integers.",
    )
    args = parser.parse_args()

    if args.clip_limit <= 0:
        raise ValueError("--clip-limit must be positive")
    if args.tile_grid_size[0] <= 0 or args.tile_grid_size[1] <= 0:
        raise ValueError("--tile-grid-size values must be positive")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    increase_contrast(args.input, args.output, args.clip_limit, tuple(args.tile_grid_size))
