from pathlib import Path

import cv2


SUPPORTED_VIDEO_EXTS = (".mp4", ".avi", ".mts", ".mov")


def is_supported_video(path):
    return Path(path).suffix.lower() in SUPPORTED_VIDEO_EXTS


def list_supported_videos(video_folder):
    root = Path(video_folder)
    if not root.exists():
        return []

    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_file() and is_supported_video(path)
        ),
        key=lambda path: path.name.lower(),
    )


def resolve_selected_videos(video_folder, video_name=None, video_file=None):
    videos = list_supported_videos(video_folder)

    if video_file:
        exact_matches = [path for path in videos if path.name == video_file]
        if exact_matches:
            return exact_matches

        lower_matches = [path for path in videos if path.name.lower() == video_file.lower()]
        if lower_matches:
            return lower_matches

        raise FileNotFoundError(f"Video file '{video_file}' not found in {video_folder}")

    if video_name:
        matches = [path for path in videos if path.stem == video_name]
        if matches:
            return matches

        lower_matches = [path for path in videos if path.stem.lower() == video_name.lower()]
        if lower_matches:
            return lower_matches

        raise FileNotFoundError(f"Video '{video_name}' not found in {video_folder}")

    return videos


def iterate_video_frames(video_path, target_fps=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = float(target_fps) if target_fps else 30.0

    frame_idx = 0
    frame_step = 1

    if target_fps and fps > target_fps:
        frame_step = max(int(round(fps / target_fps)), 1)

    def generator():
        nonlocal frame_idx
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_step > 1 and frame_idx % frame_step != 0:
                    frame_idx += 1
                    continue

                yield frame_idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_idx += 1
        finally:
            cap.release()

    return generator(), fps
