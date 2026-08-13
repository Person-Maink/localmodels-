#!/usr/bin/env python3
"""Manually annotate tremor peaks in MP4 videos with the Space key.

The OpenCV window must have focus for key presses to be recorded. Playback is
at half speed; recorded timestamps are converted to the original video timeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    import cv2
except ImportError:  # Allows calculation/storage tests without OpenCV installed.
    cv2 = None  # type: ignore[assignment]


SUMMARY_COLUMNS = [
    "video_filename",
    "video_path",
    "annotation_file",
    "peak_count",
    "duration_s",
    "fps",
    "mean_interval_s",
    "interval_std_s",
    "frequency_hz",
    "frequency_se_hz",
    "frequency_ci95_lower_hz",
    "frequency_ci95_upper_hz",
    "annotated_at_utc",
]
WINDOW_NAME = "Tremor peak annotator"
PLAYBACK_SPEED = 0.3
ANNOTATION_WINDOW_S = 5.0


def calculate_statistics(timestamps_s: Sequence[float]) -> dict[str, float | int | None]:
    """Calculate tremor statistics from strictly increasing peak timestamps."""
    timestamps = [float(value) for value in timestamps_s]
    if len(timestamps) < 2:
        raise ValueError("At least two peak timestamps are required.")
    if not all(math.isfinite(value) for value in timestamps):
        raise ValueError("Peak timestamps must be finite numbers.")

    intervals = [right - left for left, right in zip(timestamps, timestamps[1:])]
    if any(interval <= 0 for interval in intervals):
        raise ValueError("Peak timestamps must be strictly increasing.")

    mean_interval = statistics.fmean(intervals)
    frequency = 1.0 / mean_interval
    # Sample variation and an uncertainty estimate require at least two intervals.
    if len(intervals) < 2:
        interval_std: float | None = None
        frequency_se: float | None = None
        ci_lower: float | None = None
        ci_upper: float | None = None
    else:
        interval_std = statistics.stdev(intervals)
        mean_interval_se = interval_std / math.sqrt(len(intervals))
        frequency_se = mean_interval_se / (mean_interval**2)
        ci_lower = max(0.0, frequency - 1.96 * frequency_se)
        ci_upper = frequency + 1.96 * frequency_se

    return {
        "peak_count": len(timestamps),
        "mean_interval_s": mean_interval,
        "interval_std_s": interval_std,
        "frequency_hz": frequency,
        "frequency_se_hz": frequency_se,
        "frequency_ci95_lower_hz": ci_lower,
        "frequency_ci95_upper_hz": ci_upper,
    }


def read_accepted_videos(summary_path: Path) -> set[str]:
    if not summary_path.exists():
        return set()
    with summary_path.open(newline="", encoding="utf-8") as handle:
        return {
            row["video_path"]
            for row in csv.DictReader(handle)
            if row.get("video_path")
        }


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def save_annotation(
    input_dir: Path,
    video_path: Path,
    timestamps_s: Sequence[float],
    *,
    duration_s: float,
    fps: float,
    annotation_start_s: float = 0.0,
    annotation_end_s: float | None = None,
) -> Path:
    """Write detailed JSON then add or replace the corresponding summary row."""
    statistics_data = calculate_statistics(timestamps_s)
    summary_path = input_dir / "summary.csv"
    annotation_dir = input_dir / "annotations"
    annotation_path = annotation_dir / f"{video_path.stem}.json"
    relative_video_path = video_path.relative_to(input_dir).as_posix()
    relative_annotation_path = annotation_path.relative_to(input_dir).as_posix()
    annotated_at = datetime.now(timezone.utc).isoformat()

    record: dict[str, Any] = {
        "video_filename": video_path.name,
        "video_path": relative_video_path,
        "peak_timestamps_s": list(timestamps_s),
        "video_metadata": {"duration_s": duration_s, "fps": fps},
        "annotation_window": {
            "start_s": annotation_start_s,
            "end_s": duration_s if annotation_end_s is None else annotation_end_s,
        },
        "statistics": statistics_data,
        "timestamp_basis": "source_video_seconds_from_scaled_wall_clock",
        "playback_speed": PLAYBACK_SPEED,
        "annotated_at_utc": annotated_at,
    }
    _atomic_write_text(annotation_path, json.dumps(record, indent=2, allow_nan=False) + "\n")

    prior_rows: list[dict[str, str]] = []
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as handle:
            prior_rows = list(csv.DictReader(handle))
    row: dict[str, Any] = {
        "video_filename": video_path.name,
        "video_path": relative_video_path,
        "annotation_file": relative_annotation_path,
        "duration_s": duration_s,
        "fps": fps,
        "annotated_at_utc": annotated_at,
        **statistics_data,
    }
    rows = [old_row for old_row in prior_rows if old_row.get("video_path") != relative_video_path]
    rows.append({key: "" if row.get(key) is None else row.get(key, "") for key in SUMMARY_COLUMNS})
    rows.sort(key=lambda saved_row: str(saved_row["video_path"]).casefold())

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=input_dir, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        temporary_path = Path(handle.name)
    os.replace(temporary_path, summary_path)
    return annotation_path


def _draw_overlay(frame: Any, filename: str, elapsed_s: float, peak_count: int, message: str) -> Any:
    rendered = frame.copy()
    lines = [
        f"{filename}   video time: {elapsed_s:.2f} s   {PLAYBACK_SPEED:g}x",
        f"Peaks: {peak_count}   Space: mark peak",
        message,
    ]
    for index, line in enumerate(lines):
        cv2.putText(rendered, line, (16, 30 + index * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 255), 2, cv2.LINE_AA)
    return rendered


def annotate_video(video_path: Path) -> tuple[str, list[float], float, float, float, float]:
    """Play the centered annotation window and return its action and metadata."""
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed. Run: python -m pip install -r requirements.txt")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path.name}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError(f"Video has invalid FPS: {video_path.name}")
    duration_s = frame_count / fps if frame_count > 0 else 0.0
    annotation_duration_s = min(ANNOTATION_WINDOW_S, duration_s)
    start_frame = max(0, int(round((duration_s - annotation_duration_s) * fps / 2)))
    end_frame = min(int(round(frame_count)), start_frame + int(round(annotation_duration_s * fps)))
    if end_frame <= start_frame:
        capture.release()
        raise RuntimeError(f"Video contains no readable frames: {video_path.name}")
    annotation_start_s = start_frame / fps
    annotation_end_s = end_frame / fps
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    timestamps: list[float] = []
    start_time = time.perf_counter()
    frame_index = 0
    last_frame: Any | None = None
    while frame_index < end_frame - start_frame:
        ok, frame = capture.read()
        if not ok:
            break
        last_frame = frame
        # Each source-video frame stays onscreen longer at the configured slow speed.
        target_time = start_time + frame_index / (fps * PLAYBACK_SPEED)
        wait_ms = max(1, int((target_time - time.perf_counter()) * 1000))
        elapsed_s = min(annotation_start_s + frame_index / fps, annotation_end_s)
        cv2.imshow(WINDOW_NAME, _draw_overlay(frame, video_path.name, elapsed_s, len(timestamps), ""))
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord(" "):
            # Preserve the original-video timebase for frequency calculations.
            timestamp_s = min(
                annotation_end_s,
                annotation_start_s + (time.perf_counter() - start_time) * PLAYBACK_SPEED,
            )
            if not timestamps or timestamp_s > timestamps[-1]:
                timestamps.append(timestamp_s)
        elif key in (ord("q"), ord("Q"), 27):
            capture.release()
            return "quit", timestamps, duration_s, fps, annotation_start_s, annotation_end_s
        frame_index += 1

    capture.release()
    if last_frame is None:
        raise RuntimeError(f"Video contains no readable frames: {video_path.name}")
    message = "R: redo   Enter/N: accept   Q/Esc: quit"
    while True:
        cv2.imshow(WINDOW_NAME, _draw_overlay(last_frame, video_path.name, annotation_end_s, len(timestamps), message))
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("r"), ord("R")):
            return "retry", timestamps, duration_s, fps, annotation_start_s, annotation_end_s
        if key in (13, 10, ord("n"), ord("N")):
            if len(timestamps) >= 2:
                return "accept", timestamps, duration_s, fps, annotation_start_s, annotation_end_s
            message = "Need at least 2 peaks. R: redo   Q/Esc: quit"
        elif key in (ord("q"), ord("Q"), 27):
            return "quit", timestamps, duration_s, fps, annotation_start_s, annotation_end_s


def run(input_dir: Path) -> int:
    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    videos = sorted(input_dir.glob("*.mp4"), key=lambda path: path.name.casefold())
    if not videos:
        print(f"No .mp4 videos found in: {input_dir}", file=sys.stderr)
        return 2
    accepted = read_accepted_videos(input_dir / "summary.csv")
    pending = [video for video in videos if video.name not in accepted]
    print(f"Found {len(videos)} video(s); {len(pending)} pending.")
    try:
        for video in pending:
            while True:
                action, timestamps, duration_s, fps, annotation_start_s, annotation_end_s = annotate_video(video)
                if action == "quit":
                    return 0
                if action == "retry":
                    continue
                annotation_path = save_annotation(
                    input_dir, video, timestamps, duration_s=duration_s, fps=fps,
                    annotation_start_s=annotation_start_s, annotation_end_s=annotation_end_s,
                )
                print(f"Saved {video.name} -> {annotation_path.relative_to(input_dir)}")
                break
    finally:
        if cv2 is not None:
            cv2.destroyAllWindows()
    print("All pending videos have been annotated.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Folder containing MP4 videos")
    return run(parser.parse_args().input_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
