import argparse
import glob
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2


def str_to_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def hydra_bool(value: bool) -> str:
    return "True" if value else "False"


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


def retained_world_result_exists(chunk_dir: Path) -> bool:
    return len(list(chunk_dir.glob("**/*_world_results.npz"))) > 0


def completion_marker_exists(chunk_dir: Path) -> bool:
    return (
        (chunk_dir / "track_info.json").exists()
        and (chunk_dir / "cameras.json").exists()
        and retained_world_result_exists(chunk_dir)
    )


def find_completed_chunk(log_root: Path, video_name: str, start_idx: int, end_idx: int) -> Path | None:
    chunk_name = f"{video_name}-all-shot-0-{start_idx}-{end_idx}"
    pattern = log_root / "video-custom" / "*" / glob.escape(chunk_name)
    for candidate in sorted(Path(path) for path in glob.glob(str(pattern))):
        if completion_marker_exists(candidate):
            return candidate
    return None


def planned_intervals(
    frame_start: int,
    frame_end_override: int,
    total_frames: int,
    chunk_seconds: int,
    fps: float,
) -> list[tuple[int, int]]:
    effective_end = total_frames if frame_end_override < 0 else min(frame_end_override, total_frames)
    if frame_start < 0:
        raise ValueError(f"Invalid frame_start={frame_start}; expected >= 0")
    if frame_start >= effective_end:
        raise ValueError(
            f"Invalid frame range: start={frame_start}, end={effective_end}, total_frames={total_frames}"
        )

    if chunk_seconds <= 0:
        return [(frame_start, frame_end_override if frame_end_override >= 0 else -1)]

    chunk_frames = max(1, int(round(fps * chunk_seconds)))
    if (effective_end - frame_start) <= chunk_frames:
        return [(frame_start, frame_end_override if frame_end_override >= 0 else -1)]

    bounds: list[tuple[int, int]] = []
    start = frame_start
    while start < effective_end:
        stop = min(start + chunk_frames, effective_end)
        bounds.append((start, stop))
        start = stop
    return bounds


def effective_stop(total_frames: int, end_idx: int) -> int:
    return total_frames if end_idx < 0 else min(end_idx, total_frames)


def completion_marker_path(log_root: Path, video_name: str) -> Path:
    return log_root.parent / "_completed" / f"{video_name}.json"


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def clear_overwrite_state(log_root: Path, video_name: str) -> None:
    marker_path = completion_marker_path(log_root, video_name)
    if marker_path.exists():
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
        for retained_dir in payload.get("retained_dirs", []):
            remove_path(Path(retained_dir))
        remove_path(marker_path)

    pattern = log_root / "video-custom" / "*" / f"{video_name}-all-shot-0-*"
    for candidate in glob.glob(str(pattern)):
        remove_path(Path(candidate))


def select_world_result(chunk_dir: Path) -> Path:
    for preferred_dir in ("prior", "smooth_fit", "root_fit"):
        candidates = sorted((chunk_dir / preferred_dir).glob("*_world_results.npz"))
        if candidates:
            return candidates[-1]

    fallback = sorted(chunk_dir.glob("**/*_world_results.npz"))
    if not fallback:
        raise FileNotFoundError(f"No *_world_results.npz file found under {chunk_dir}")
    return fallback[-1]


def prune_chunk_outputs(chunk_dir: Path) -> Path:
    cameras_path = chunk_dir / "cameras.json"
    track_info_path = chunk_dir / "track_info.json"
    if not cameras_path.exists() or not track_info_path.exists():
        raise FileNotFoundError(f"Missing cameras.json or track_info.json in {chunk_dir}")

    world_result = select_world_result(chunk_dir)
    retained_world_path = chunk_dir / world_result.name
    if world_result != retained_world_path:
        shutil.copy2(world_result, retained_world_path)

    for child in chunk_dir.iterdir():
        if child == cameras_path or child == track_info_path or child == retained_world_path:
            continue
        remove_path(child)

    return retained_world_path


def write_completion_marker(log_root: Path, video_name: str, retained_dirs: list[Path]) -> None:
    marker_path = completion_marker_path(log_root, video_name)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "dynhamr",
        "video": video_name,
        "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "retained_dirs": [str(path) for path in retained_dirs],
    }
    marker_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_run_command(
    args: argparse.Namespace,
    start_idx: int,
    end_idx: int,
    data_root: Path,
    hmp_frame_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(args.run_opt_script),
        "data=video",
        "run_opt=True",
        f"run_vis={hydra_bool(args.run_vis)}",
        f"run_prior={hydra_bool(args.run_prior)}",
        f"data.root={data_root}",
        "data.video_dir=images",
        f"data.seq='{args.video_name}'",
        f"data.ext={args.video_ext}",
        f"data.src_path='{args.video}'",
        f"data.start_idx={start_idx}",
        f"data.end_idx={end_idx}",
        f"is_static={hydra_bool(args.is_static)}",
        f"temporal_smooth={hydra_bool(args.temporal_smooth)}",
        f"HMP.vid_path='{hmp_frame_dir}'",
        f"optim.root.num_iters={args.root_iters}",
        f"optim.smooth.num_iters={args.smooth_iters}",
        f"log_root='{args.log_root}'",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Dyn-HaMR on a selected video interval in sequential chunks to keep memory bounded."
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--data-root", required=False, type=Path, default=None)
    parser.add_argument("--video-dir", required=False, default="images")
    parser.add_argument("--video-name", required=True)
    parser.add_argument("--video-ext", required=True)
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--hmp-frame-dir", required=False, type=Path, default=None)
    parser.add_argument("--temp-parent", required=False, type=Path, default=None)
    parser.add_argument("--run-vis", type=str_to_bool, default=False)
    parser.add_argument("--run-prior", type=str_to_bool, default=False)
    parser.add_argument("--is-static", type=str_to_bool, default=False)
    parser.add_argument("--temporal-smooth", type=str_to_bool, default=False)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=-1)
    parser.add_argument("--chunk-seconds", type=int, default=600)
    parser.add_argument("--root-iters", type=int, default=40)
    parser.add_argument("--smooth-iters", type=int, default=60)
    parser.add_argument("--skip-existing", type=str_to_bool, default=True)
    parser.add_argument("--overwrite", type=str_to_bool, default=False)
    parser.add_argument("--keep-temp-data", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-opt-script", type=Path, default=Path(__file__).resolve().parent / "dyn-hamr" / "run_opt.py")
    args = parser.parse_args()

    args.video = args.video.resolve()
    args.log_root = args.log_root.resolve()
    args.run_opt_script = args.run_opt_script.resolve()

    if args.temp_parent is None:
        args.temp_parent = Path(tempfile.gettempdir())
    else:
        args.temp_parent = args.temp_parent.resolve()
    args.temp_parent.mkdir(parents=True, exist_ok=True)

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.run_opt_script.exists():
        raise FileNotFoundError(f"run_opt.py not found: {args.run_opt_script}")

    marker_path = completion_marker_path(args.log_root, args.video_name)
    if marker_path.exists() and not args.overwrite:
        print(f"Skipping {args.video_name}: completion marker already exists at {marker_path}")
        return

    if args.overwrite:
        print(f"Overwrite requested for {args.video_name}; clearing retained Dyn-HaMR outputs.")
        clear_overwrite_state(args.log_root, args.video_name)

    fps, total_frames = video_metadata(args.video)
    intervals = planned_intervals(args.start_idx, args.end_idx, total_frames, args.chunk_seconds, fps)
    effective_end = effective_stop(total_frames, args.end_idx)
    chunk_frames = (
        max(1, int(round(fps * args.chunk_seconds))) if args.chunk_seconds > 0 else (effective_end - args.start_idx)
    )

    print(f"Video: {args.video}")
    print(
        f"Requested Dyn-HaMR interval: [{args.start_idx}, "
        f"{args.end_idx if args.end_idx >= 0 else effective_end})"
    )
    print(f"Video metadata: fps={fps:.3f}, total_frames={total_frames}")
    if args.chunk_seconds > 0:
        print(
            f"Chunking enabled: {len(intervals)} slice(s), up to {args.chunk_seconds} seconds "
            f"({chunk_frames} frame(s)) each."
        )
    else:
        print("Chunking disabled because chunk_seconds <= 0; running the requested interval as one slice.")
    print(f"Chunk reuse is {'enabled' if args.skip_existing else 'disabled'}.")
    print(f"Temporary Dyn-HaMR data root parent: {args.temp_parent}")

    retained_dirs: list[Path] = []
    completed = 0

    for index, (chunk_start, chunk_end) in enumerate(intervals, start=1):
        display_end = effective_stop(total_frames, chunk_end)

        if args.skip_existing:
            completed_dir = find_completed_chunk(args.log_root, args.video_name, chunk_start, chunk_end)
            if completed_dir is not None:
                prune_chunk_outputs(completed_dir)
                print(
                    f"[{index}/{len(intervals)}] Reusing completed chunk "
                    f"[{chunk_start}, {display_end}) from {completed_dir}"
                )
                retained_dirs.append(completed_dir)
                completed += 1
                continue

        chunk_data_root = Path(
            tempfile.mkdtemp(
                prefix=f"dynhamr_{args.video_name}_{chunk_start}_{display_end}_",
                dir=str(args.temp_parent),
            )
        )
        hmp_frame_dir = chunk_data_root / "images" / args.video_name
        cmd = build_run_command(args, chunk_start, chunk_end, chunk_data_root, hmp_frame_dir)
        print(f"[{index}/{len(intervals)}] Running chunk [{chunk_start}, {display_end})")
        print(f"Temporary data root: {chunk_data_root}")
        print(f"Command: {' '.join(cmd)}")

        try:
            if not args.dry_run:
                subprocess.run(cmd, cwd=args.run_opt_script.parent, check=True)
                completed_dir = find_completed_chunk(args.log_root, args.video_name, chunk_start, chunk_end)
                if completed_dir is None:
                    raise FileNotFoundError(
                        f"Could not locate completed Dyn-HaMR chunk output for [{chunk_start}, {display_end})"
                    )
                prune_chunk_outputs(completed_dir)
                retained_dirs.append(completed_dir)
                completed += 1
        finally:
            if args.keep_temp_data:
                print(f"Keeping temporary Dyn-HaMR data root for debugging: {chunk_data_root}")
            else:
                remove_path(chunk_data_root)

    if args.dry_run:
        print("Dry run complete; no Dyn-HaMR chunks were executed.")
        return

    write_completion_marker(args.log_root, args.video_name, retained_dirs)
    print(f"Finished {completed}/{len(intervals)} Dyn-HaMR chunk(s) for {args.video_name}.")
    print(f"Dyn-HaMR artifacts root: {args.log_root}")


if __name__ == "__main__":
    main()
