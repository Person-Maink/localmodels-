import cv2
import argparse
import os
from natsort import natsorted

def images_to_video(input_folder, output_path, fps=30):
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images = natsorted(images, key=lambda x: os.path.splitext(x)[0])

    if not images:
        raise ValueError(f"No images found in folder: {input_folder}")

    first_frame = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_name in images:
        img_path = os.path.join(input_folder, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Skipping unreadable image {img_name}")
            continue
        out.write(frame)

    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine two videos side by side.")
    parser.add_argument("--frames", type=str, default="/home/mayank/Documents/Uni/TUD/Thesis Extra/DELFTBLUE /frames/",  help="Path to frames for the video")
    parser.add_argument("--output", type=str, default="/home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/other scripts/output/video.mp4", help="Output file path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    images_to_video(args.frames, args.output)


