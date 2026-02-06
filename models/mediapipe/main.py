import argparse
from inference import * 
from loader import *
from visualize import *
import os

def main(args):
    os.makedirs(args.output_folder, exist_ok=True)

    for fname in os.listdir(args.video_folder):

        video_path = os.path.join(args.video_folder, fname)
        base = os.path.splitext(fname)[0]
        out_csv = os.path.join(args.output_folder + "keypoints/", f"{base}_keypoints.csv")
        out_video = os.path.join(args.output_folder + "visualizations/", f"{base}_overlay.mp4")

        print(f"\nProcessing: {fname}")
        
        if not os.path.exists(out_csv):
            frame_iterator, fps = iterate_video_frames(video_path, target_fps=args.target_fps)
            df = mediapipe_inference(frame_iterator, fps)

            df.to_csv(out_csv, index=False)
            print(f"Saved CSV: {out_csv}")
        else: 
            print("Keypoints already exist. Skipping inference.")

        if args.visualize:
            overlay_hand_pose(video_path, out_csv, out_video)
            print(f"Saved visualization: {out_video}")

    print("\nAll videos processed.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run Mediapipe hand inference on all videos in a folder."
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        default="../../data/images/",
        help="Path to folder containing input videos."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="../../outputs/mediapipe/",
        help="Folder where output CSVs and visualizations are saved."
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="If set, overlay joints and save visualization video."
    )
    parser.add_argument(
        "--target_fps",
        type=int,
        default=30,
        help="Target frame rate for inference."
    )

    args = parser.parse_args()
    main(args)
