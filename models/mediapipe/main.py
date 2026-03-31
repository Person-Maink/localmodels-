import argparse
from pathlib import Path

from inference import mediapipe_inference
from loader import resolve_selected_videos
from loader import iterate_video_frames
from loader import list_supported_videos
from visualize import overlay_hand_pose


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Mediapipe hand inference on videos in a folder.",
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        default="../../data/images/",
        help="Path to folder containing input videos.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Video stem to process, for example 'clip_01'.",
    )
    parser.add_argument(
        "--video_file",
        type=str,
        default=None,
        help="Exact video filename to process, for example 'clip_01.MTS'.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="../../outputs/mediapipe/",
        help="Folder where output CSVs and visualizations are saved.",
    )
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to overlay joints and save visualization videos.",
    )
    parser.add_argument(
        "--target_fps",
        type=int,
        default=30,
        help="Target frame rate for inference.",
    )
    return parser.parse_args()


def format_available_videos(video_folder):
    videos = list_supported_videos(video_folder)
    if not videos:
        return "none"
    return ", ".join(path.stem for path in videos)


def run_video(video_path, output_root, target_fps, visualize):
    base_name = video_path.stem
    keypoints_dir = output_root / "keypoints"
    visualizations_dir = output_root / "visualizations"
    keypoints_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    out_csv = keypoints_dir / f"{base_name}_keypoints.csv"
    out_video = visualizations_dir / f"{base_name}_overlay.mp4"

    print(f"\nProcessing: {video_path.name}")

    if out_csv.exists():
        print(f"Keypoints already exist. Skipping inference: {out_csv}")
    else:
        frame_iterator, fps = iterate_video_frames(video_path, target_fps=target_fps)
        df = mediapipe_inference(frame_iterator, fps)
        df.to_csv(out_csv, index=False)
        print(f"Saved CSV: {out_csv}")

    if visualize:
        overlay_hand_pose(str(video_path), str(out_csv), str(out_video))
        print(f"Saved visualization: {out_video}")


def main(args):
    output_root = Path(args.output_folder)
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        videos = resolve_selected_videos(
            args.video_folder,
            video_name=args.video,
            video_file=args.video_file,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\nAvailable videos: {format_available_videos(args.video_folder)}"
        ) from exc

    if not videos:
        raise SystemExit(
            f"No supported videos found in {args.video_folder}. "
            "Expected one of: mp4, avi, mts, mov."
        )

    for video_path in videos:
        run_video(video_path, output_root, args.target_fps, args.visualize)

    print("\nAll videos processed.")


if __name__ == "__main__":
    main(parse_args())
