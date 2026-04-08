import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def str_to_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def video_metadata(video_path: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0:
        raise RuntimeError(f"Invalid FPS for video: {video_path}")
    if total_frames <= 0:
        raise RuntimeError(f"Invalid frame count for video: {video_path}")
    return fps, total_frames


def chunk_bounds(frame_start: int, frame_end: int, frame_skip: int, fps: float, chunk_seconds: int) -> list[tuple[int, int]]:
    chunk_frames = max(frame_skip, int(round(fps * chunk_seconds)))
    bounds: list[tuple[int, int]] = []
    start = frame_start
    while start < frame_end:
        stop = min(start + chunk_frames, frame_end)
        bounds.append((start, stop))
        start = stop
    return bounds


def camera_outputs_exist(output_root: Path, artifact_name: str) -> bool:
    return (
        (output_root / "pose" / f"{artifact_name}.npz").exists()
        and (output_root / "intrinsics" / f"{artifact_name}.npz").exists()
    )


def chunk_root(output_root: Path, artifact_name: str, chunk_start: int, chunk_end: int) -> Path:
    return output_root / "_chunks" / artifact_name / f"{chunk_start:09d}_{chunk_end:09d}"


def run_chunk(
    model_root: Path,
    video_path: Path,
    output_root: Path,
    pipeline: str,
    frame_start: int,
    frame_end: int,
    frame_skip: int,
    save_viz: bool,
) -> None:
    chunk_output = chunk_root(output_root, video_path.stem, frame_start, frame_end)
    cmd = [
        sys.executable,
        "run.py",
        f"pipeline={pipeline}",
        "streams=raw_mp4_stream",
        f"streams.base_path={video_path}",
        f"streams.frame_start={frame_start}",
        f"streams.frame_end={frame_end}",
        f"streams.frame_skip={frame_skip}",
        f"pipeline.output.path={chunk_output}",
        "pipeline.output.skip_exists=true",
        "pipeline.output.save_camera_artifacts=true",
        "pipeline.output.save_artifacts=false",
        f"pipeline.output.save_viz={'true' if save_viz else 'false'}",
    ]
    subprocess.run(cmd, cwd=model_root, check=True)


def merge_chunk_outputs(output_root: Path, artifact_name: str, chunks: list[tuple[int, int]]) -> None:
    pose_inds_all = []
    pose_data_all = []
    intr_inds_all = []
    intr_data_all = []

    for chunk_start, chunk_end in chunks:
        chunk_output = chunk_root(output_root, artifact_name, chunk_start, chunk_end)
        pose_path = chunk_output / "pose" / f"{artifact_name}.npz"
        intr_path = chunk_output / "intrinsics" / f"{artifact_name}.npz"
        if not pose_path.exists() or not intr_path.exists():
            raise FileNotFoundError(
                f"Missing chunk outputs for {artifact_name} frames [{chunk_start}, {chunk_end}): {chunk_output}"
            )

        pose_chunk = np.load(pose_path)
        intr_chunk = np.load(intr_path)
        pose_inds_all.append(np.asarray(pose_chunk["inds"], dtype=np.int64))
        pose_data_all.append(np.asarray(pose_chunk["data"], dtype=np.float32))
        intr_inds_all.append(np.asarray(intr_chunk["inds"], dtype=np.int64))
        intr_data_all.append(np.asarray(intr_chunk["data"], dtype=np.float32))

    pose_inds = np.concatenate(pose_inds_all, axis=0)
    pose_data = np.concatenate(pose_data_all, axis=0)
    intr_inds = np.concatenate(intr_inds_all, axis=0)
    intr_data = np.concatenate(intr_data_all, axis=0)

    pose_order = np.argsort(pose_inds)
    intr_order = np.argsort(intr_inds)
    pose_inds = pose_inds[pose_order]
    pose_data = pose_data[pose_order]
    intr_inds = intr_inds[intr_order]
    intr_data = intr_data[intr_order]

    pose_inds, pose_unique = np.unique(pose_inds, return_index=True)
    intr_inds, intr_unique = np.unique(intr_inds, return_index=True)
    pose_data = pose_data[pose_unique]
    intr_data = intr_data[intr_unique]

    pose_out = output_root / "pose" / f"{artifact_name}.npz"
    intr_out = output_root / "intrinsics" / f"{artifact_name}.npz"
    cam_type_out = output_root / "intrinsics" / f"{artifact_name}_camera.txt"

    pose_out.parent.mkdir(parents=True, exist_ok=True)
    intr_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(pose_out, data=pose_data, inds=pose_inds)
    np.savez(intr_out, data=intr_data, inds=intr_inds)

    with cam_type_out.open("w") as f:
        for frame_idx in intr_inds:
            f.write(f"{int(frame_idx)}: PINHOLE\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ViPE in memory-friendlier chunks and merge camera outputs.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pipeline", default="no_vda")
    parser.add_argument("--chunk-seconds", type=int, default=600)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=-1)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--skip-existing", type=str_to_bool, default=True)
    parser.add_argument("--save-viz", type=str_to_bool, default=False)
    args = parser.parse_args()

    model_root = Path(__file__).resolve().parent
    video_path = args.video.resolve()
    output_root = args.output.resolve()
    artifact_name = video_path.stem

    if args.skip_existing and camera_outputs_exist(output_root, artifact_name):
        print(f"Skipping {artifact_name}: final pose/intrinsics already exist under {output_root}")
        return

    fps, total_frames = video_metadata(video_path)
    frame_end = total_frames if args.frame_end == -1 else min(args.frame_end, total_frames)
    if args.frame_start < 0 or args.frame_start >= frame_end:
        raise ValueError(
            f"Invalid frame range for {video_path}: start={args.frame_start}, end={frame_end}, total={total_frames}"
        )

    chunks = chunk_bounds(args.frame_start, frame_end, args.frame_skip, fps, args.chunk_seconds)
    print(
        f"Processing {artifact_name} in {len(chunks)} chunk(s) of up to {args.chunk_seconds} seconds "
        f"from frame {args.frame_start} to {frame_end}."
    )

    for chunk_start, chunk_end in chunks:
        current_chunk_root = chunk_root(output_root, artifact_name, chunk_start, chunk_end)
        if args.skip_existing and camera_outputs_exist(current_chunk_root, artifact_name):
            print(f"Skipping chunk [{chunk_start}, {chunk_end}) for {artifact_name}: outputs already exist")
            continue

        print(f"Running chunk [{chunk_start}, {chunk_end}) for {artifact_name}")
        run_chunk(
            model_root=model_root,
            video_path=video_path,
            output_root=output_root,
            pipeline=args.pipeline,
            frame_start=chunk_start,
            frame_end=chunk_end,
            frame_skip=args.frame_skip,
            save_viz=args.save_viz,
        )

    merge_chunk_outputs(output_root, artifact_name, chunks)
    print(f"Merged pose/intrinsics for {artifact_name} into {output_root}")


if __name__ == "__main__":
    main()
