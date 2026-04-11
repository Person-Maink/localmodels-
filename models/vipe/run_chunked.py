import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
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


def align_chunk_frames(chunk_frames: int, frame_skip: int) -> int:
    aligned = (chunk_frames // frame_skip) * frame_skip
    return max(frame_skip, aligned)


def chunk_bounds(
    frame_start: int,
    frame_end: int,
    frame_skip: int,
    fps: float,
    chunk_seconds: int,
    max_chunk_frames: int,
) -> list[tuple[int, int]]:
    chunk_frames = max(frame_skip, int(round(fps * chunk_seconds)))
    if max_chunk_frames > 0:
        chunk_frames = min(chunk_frames, align_chunk_frames(max_chunk_frames, frame_skip))
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


def completion_marker_path(output_root: Path, artifact_name: str) -> Path:
    return output_root / "_completed" / f"{artifact_name}.json"


def remove_if_exists(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def clear_video_outputs(output_root: Path, artifact_name: str) -> None:
    remove_if_exists(output_root / "pose" / f"{artifact_name}.npz")
    remove_if_exists(output_root / "intrinsics" / f"{artifact_name}.npz")
    remove_if_exists(output_root / "intrinsics" / f"{artifact_name}_camera.txt")
    remove_if_exists(output_root / "_chunks" / artifact_name)
    remove_if_exists(completion_marker_path(output_root, artifact_name))


def split_chunk(frame_start: int, frame_end: int, frame_skip: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    chunk_frames = frame_end - frame_start
    if chunk_frames <= frame_skip:
        return None

    midpoint = frame_start + (chunk_frames // 2)
    midpoint = frame_start + align_chunk_frames(midpoint - frame_start, frame_skip)
    if midpoint <= frame_start:
        midpoint = frame_start + frame_skip
    if midpoint >= frame_end:
        midpoint = frame_end - frame_skip
    if midpoint <= frame_start or midpoint >= frame_end:
        return None

    return (frame_start, midpoint), (midpoint, frame_end)


def write_completion_marker(output_root: Path, artifact_name: str) -> None:
    marker_path = completion_marker_path(output_root, artifact_name)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "vipe",
        "video": artifact_name,
        "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "retained_paths": [
            str(output_root / "pose" / f"{artifact_name}.npz"),
            str(output_root / "intrinsics" / f"{artifact_name}.npz"),
            str(output_root / "intrinsics" / f"{artifact_name}_camera.txt"),
        ],
    }
    marker_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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


def process_chunk(
    model_root: Path,
    video_path: Path,
    output_root: Path,
    pipeline: str,
    frame_start: int,
    frame_end: int,
    frame_skip: int,
    save_viz: bool,
    skip_existing: bool,
    min_chunk_frames: int,
    depth: int = 0,
) -> list[tuple[int, int]]:
    artifact_name = video_path.stem
    current_chunk_root = chunk_root(output_root, artifact_name, frame_start, frame_end)
    chunk_frames = frame_end - frame_start
    indent = "  " * depth

    if skip_existing and camera_outputs_exist(current_chunk_root, artifact_name):
        print(f"{indent}Skipping chunk [{frame_start}, {frame_end}) for {artifact_name}: outputs already exist")
        return [(frame_start, frame_end)]

    remove_if_exists(current_chunk_root)
    print(f"{indent}Running chunk [{frame_start}, {frame_end}) for {artifact_name} ({chunk_frames} frames)")

    try:
        run_chunk(
            model_root=model_root,
            video_path=video_path,
            output_root=output_root,
            pipeline=pipeline,
            frame_start=frame_start,
            frame_end=frame_end,
            frame_skip=frame_skip,
            save_viz=save_viz,
        )
        return [(frame_start, frame_end)]
    except subprocess.CalledProcessError as exc:
        remove_if_exists(current_chunk_root)
        split_bounds = split_chunk(frame_start, frame_end, frame_skip)
        if chunk_frames <= min_chunk_frames or split_bounds is None:
            print(
                f"{indent}Chunk [{frame_start}, {frame_end}) for {artifact_name} failed with exit code "
                f"{exc.returncode} and cannot be split further."
            )
            raise

        left_bounds, right_bounds = split_bounds
        print(
            f"{indent}Chunk [{frame_start}, {frame_end}) for {artifact_name} failed with exit code "
            f"{exc.returncode}; retrying as [{left_bounds[0]}, {left_bounds[1]}) and "
            f"[{right_bounds[0]}, {right_bounds[1]})."
        )
        processed_chunks: list[tuple[int, int]] = []
        processed_chunks.extend(
            process_chunk(
                model_root=model_root,
                video_path=video_path,
                output_root=output_root,
                pipeline=pipeline,
                frame_start=left_bounds[0],
                frame_end=left_bounds[1],
                frame_skip=frame_skip,
                save_viz=save_viz,
                skip_existing=skip_existing,
                min_chunk_frames=min_chunk_frames,
                depth=depth + 1,
            )
        )
        processed_chunks.extend(
            process_chunk(
                model_root=model_root,
                video_path=video_path,
                output_root=output_root,
                pipeline=pipeline,
                frame_start=right_bounds[0],
                frame_end=right_bounds[1],
                frame_skip=frame_skip,
                save_viz=save_viz,
                skip_existing=skip_existing,
                min_chunk_frames=min_chunk_frames,
                depth=depth + 1,
            )
        )
        return processed_chunks


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
    parser.add_argument("--chunk-seconds", type=int, default=300)
    parser.add_argument("--max-chunk-frames", type=int, default=960)
    parser.add_argument("--min-chunk-frames", type=int, default=256)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=-1)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--skip-existing", type=str_to_bool, default=True)
    parser.add_argument("--save-viz", type=str_to_bool, default=False)
    parser.add_argument("--overwrite", type=str_to_bool, default=False)
    args = parser.parse_args()

    model_root = Path(__file__).resolve().parent
    video_path = args.video.resolve()
    output_root = args.output.resolve()
    artifact_name = video_path.stem
    marker_path = completion_marker_path(output_root, artifact_name)

    if marker_path.exists() and not args.overwrite:
        print(f"Skipping {artifact_name}: completion marker already exists at {marker_path}")
        return

    if args.overwrite:
        print(f"Overwrite requested for {artifact_name}; clearing retained outputs under {output_root}")
        clear_video_outputs(output_root, artifact_name)

    fps, total_frames = video_metadata(video_path)
    frame_end = total_frames if args.frame_end == -1 else min(args.frame_end, total_frames)
    if args.frame_start < 0 or args.frame_start >= frame_end:
        raise ValueError(
            f"Invalid frame range for {video_path}: start={args.frame_start}, end={frame_end}, total={total_frames}"
        )
    if args.min_chunk_frames < args.frame_skip:
        raise ValueError(
            f"min_chunk_frames must be at least frame_skip: min_chunk_frames={args.min_chunk_frames}, "
            f"frame_skip={args.frame_skip}"
        )

    chunks = chunk_bounds(
        args.frame_start,
        frame_end,
        args.frame_skip,
        fps,
        args.chunk_seconds,
        args.max_chunk_frames,
    )
    print(
        f"Processing {artifact_name} in {len(chunks)} chunk(s) of up to {args.chunk_seconds} seconds "
        f"and {args.max_chunk_frames} frames from frame {args.frame_start} to {frame_end}."
    )

    processed_chunks: list[tuple[int, int]] = []
    for chunk_start, chunk_end in chunks:
        processed_chunks.extend(
            process_chunk(
                model_root=model_root,
                video_path=video_path,
                output_root=output_root,
                pipeline=args.pipeline,
                frame_start=chunk_start,
                frame_end=chunk_end,
                frame_skip=args.frame_skip,
                save_viz=args.save_viz,
                skip_existing=args.skip_existing,
                min_chunk_frames=args.min_chunk_frames,
            )
        )

    merge_chunk_outputs(output_root, artifact_name, processed_chunks)
    remove_if_exists(output_root / "_chunks" / artifact_name)
    write_completion_marker(output_root, artifact_name)
    print(f"Merged pose/intrinsics for {artifact_name} into {output_root}")


if __name__ == "__main__":
    main()
