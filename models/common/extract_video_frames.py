import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a video interval into numbered JPG frames.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    args = parser.parse_args()

    if args.start_frame < 0:
        raise ValueError("--start-frame must be >= 0")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = total_frames if args.end_frame < 0 else min(args.end_frame, total_frames)
    if args.start_frame >= end_frame:
        raise ValueError(
            f"Invalid frame range for {args.video}: start={args.start_frame}, end={end_frame}, total={total_frames}"
        )

    if not cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame):
        cap.release()
        raise RuntimeError(f"Failed to seek to frame {args.start_frame} in {args.video}")

    frame_idx = args.start_frame
    written = 0
    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        out_path = args.output_dir / f"frame_{frame_idx:06d}.jpg"
        if not cv2.imwrite(str(out_path), frame):
            cap.release()
            raise RuntimeError(f"Failed to write frame to {out_path}")

        written += 1
        frame_idx += 1

    cap.release()

    if written == 0:
        raise RuntimeError(f"No frames were extracted from {args.video}")

    print(f"Extracted {written} frame(s) from {args.video} into {args.output_dir}")


if __name__ == "__main__":
    main()
