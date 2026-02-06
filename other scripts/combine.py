import cv2
import argparse
import os

def combine_side_by_side(video1_path, video2_path, output_path):
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not cap1.isOpened() or not cap2.isOpened():
        raise RuntimeError("One of the input videos could not be opened.")

    # get basic properties from the first video
    fps = cap1.get(cv2.CAP_PROP_FPS)
    width  = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # output: double width (side-by-side)
    out_width = width * 2
    out_height = height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    print(f"Saving combined video to: {output_path}")
    print(f"Resolution: {out_width}x{out_height} | FPS: {fps}")

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break

        # ensure equal size (just in case)
        frame2 = cv2.resize(frame2, (width, height))

        combined = cv2.hconcat([frame1, frame2])
        out.write(combined)

    cap1.release()
    cap2.release()
    out.release()
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine two videos side by side.")
    parser.add_argument("--video1", type=str, default="/home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/outputs/wilor/videos/combined.mp4",  help="Path to first video (left)")
    parser.add_argument("--video2", type=str, default="/home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/data/images/163 (2) FU.MTS",  help="Path to second video (right)")
    parser.add_argument("--output", type=str, default="/home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/outputs/wilor/videos/combined_side_by_side.mp4", help="Output file path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    combine_side_by_side(args.video1, args.video2, args.output)
