#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from npy_io import list_frame_folders, load_frame_records


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_CLIP = "me 1"
DEFAULT_FPS = 30
DEFAULT_START = 0.0
DEFAULT_DURATION = 3.0
DEFAULT_PANEL_WIDTH = 480
DEFAULT_OUTPUT = PROJECT_ROOT / "assets" / "readme" / "me1_pipeline_montage.gif"

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a GitHub-friendly README GIF montage from existing pipeline videos and analysis outputs."
    )
    parser.add_argument("--clip", default=DEFAULT_CLIP, help="Clip name shared by the input videos.")
    parser.add_argument("--start", type=float, default=DEFAULT_START, help="Start time in seconds.")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION, help="Duration in seconds.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Output GIF frame rate.")
    parser.add_argument(
        "--panel-width",
        type=int,
        default=DEFAULT_PANEL_WIDTH,
        help="Width of each panel in pixels before stacking.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output GIF path.",
    )
    return parser.parse_args()


def require_tool(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise RuntimeError(f"Required tool not found on PATH: {name}")
    return resolved


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def probe_duration(ffprobe_bin: str, video_path: Path) -> float:
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    duration_text = result.stdout.strip()
    if not duration_text:
        raise RuntimeError(f"Unable to read duration for {video_path}")
    return float(duration_text)


def shlex_quote(value: str) -> str:
    if value == "":
        return "''"
    if all(ch.isalnum() or ch in "@%_+=:,./-" for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def drawtext_filter(label: str) -> str:
    safe = label.replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")
    return (
        "drawtext="
        f"text='{safe}':"
        "fontcolor=white:"
        "fontsize=28:"
        "x=20:"
        "y=20:"
        "box=1:"
        "boxcolor=black@0.6:"
        "boxborderw=12"
    )


def _pick_primary_record(records: list[dict]) -> dict:
    if not records:
        raise ValueError("Cannot pick a primary record from an empty list.")
    return sorted(
        records,
        key=lambda rec: (
            -1 if rec["box_size"] is None else -float(rec["box_size"]),
            str(rec["path"]),
        ),
    )[0]


def _load_grounded_wilor_frames(frames_root: Path, start_frame: int, end_frame: int) -> list[dict]:
    folders = list_frame_folders(str(frames_root))
    selected = folders[start_frame:end_frame]
    frames: list[dict] = []

    for folder in selected:
        records = load_frame_records(folder, pattern="*.npy")
        if not records:
            continue

        rec = _pick_primary_record(records)
        verts_world = rec["verts_world"]
        arr = np.load(rec["path"], allow_pickle=True).item()
        joints = np.asarray(arr["joints"], dtype=np.float32) + rec["cam_t"].reshape(1, 3)
        wrist = joints[0]

        frames.append(
            {
                "verts": verts_world - wrist,
                "joints": joints - wrist,
                "right": int(rec["right"]),
            }
        )

    if not frames:
        raise RuntimeError(f"No grounded WiLoR frames found in: {frames_root}")
    return frames


def _compute_bounds(frames: list[dict]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    verts = np.concatenate([frame["verts"][::2] for frame in frames], axis=0)
    joints = np.concatenate([frame["joints"] for frame in frames], axis=0)
    all_points = np.concatenate([verts, joints], axis=0)

    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    ranges = maxs - mins
    pad = np.maximum(ranges * 0.18, 0.02)

    return (
        (float(mins[0] - pad[0]), float(maxs[0] + pad[0])),
        (float(mins[1] - pad[1]), float(maxs[1] + pad[1])),
        (float(mins[2] - pad[2]), float(maxs[2] + pad[2])),
    )


def _hand_color(right_flag: int) -> str:
    return "#ef4444" if right_flag == 1 else "#60a5fa"


def _render_grounded_panel_pngs(
    frames: list[dict],
    out_dir: Path,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> None:
    xlim, ylim, zlim = bounds
    grid_x = np.linspace(xlim[0], xlim[1], 7)
    grid_y = np.linspace(ylim[0], ylim[1], 7)
    ground_z = zlim[0] + (zlim[1] - zlim[0]) * 0.03

    for idx, frame in enumerate(frames):
        fig = plt.figure(figsize=(16, 9), dpi=120)
        fig.patch.set_facecolor("#08111b")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#08111b")

        verts = frame["verts"]
        joints = frame["joints"]
        color = _hand_color(frame["right"])

        for gx in grid_x:
            ax.plot([gx, gx], [grid_y[0], grid_y[-1]], [ground_z, ground_z], color="#1f3347", lw=0.8, alpha=0.75)
        for gy in grid_y:
            ax.plot([grid_x[0], grid_x[-1]], [gy, gy], [ground_z, ground_z], color="#1f3347", lw=0.8, alpha=0.75)

        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=3, c=color, alpha=0.22, linewidths=0)
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=22, c="#f8fafc", alpha=0.95, depthshade=False)

        for a, b in HAND_CONNECTIONS:
            ax.plot(
                [joints[a, 0], joints[b, 0]],
                [joints[a, 1], joints[b, 1]],
                [joints[a, 2], joints[b, 2]],
                color=color,
                lw=2.4,
                alpha=0.95,
            )

        ax.scatter([0.0], [0.0], [0.0], s=46, c="#facc15", depthshade=False)
        ax.text2D(
            0.04,
            0.92,
            "Wrist-grounded WiLoR",
            transform=ax.transAxes,
            color="#f8fafc",
            fontsize=18,
            weight="bold",
        )
        ax.text2D(
            0.04,
            0.875,
            "Analysis-space reconstruction",
            transform=ax.transAxes,
            color="#9fb3c8",
            fontsize=11,
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.view_init(elev=22, azim=-58)
        ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], (zlim[1] - zlim[0]) * 0.9))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.line.set_color((0, 0, 0, 0))
        ax.yaxis.line.set_color((0, 0, 0, 0))
        ax.zaxis.line.set_color((0, 0, 0, 0))

        out_path = out_dir / f"frame_{idx:06d}.png"
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches=None, pad_inches=0)
        plt.close(fig)


def _make_grounded_wilor_video(
    ffmpeg_bin: str,
    frames_root: Path,
    temp_dir: Path,
    args: argparse.Namespace,
) -> Path:
    start_frame = max(0, int(round(args.start * args.fps)))
    frame_count = max(1, int(round(args.duration * args.fps)))
    end_frame = start_frame + frame_count

    frames = _load_grounded_wilor_frames(frames_root, start_frame, end_frame)
    if len(frames) < frame_count:
        frame_count = len(frames)
        frames = frames[:frame_count]

    frame_dir = temp_dir / "grounded_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    bounds = _compute_bounds(frames)
    _render_grounded_panel_pngs(frames, frame_dir, bounds)

    analysis_mp4 = temp_dir / "grounded_wilor_analysis.mp4"
    command = [
        ffmpeg_bin,
        "-y",
        "-framerate",
        str(args.fps),
        "-i",
        str(frame_dir / "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(analysis_mp4),
    ]
    subprocess.run(command, check=True)
    return analysis_mp4


def build_filter(args: argparse.Namespace) -> str:
    labels = ("Input", "WiLoR", "Grounded WiLoR")
    chains = []
    for idx, label in enumerate(labels):
        chains.append(
            f"[{idx}:v]"
            f"trim=start={args.start}:duration={args.duration},"
            "setpts=PTS-STARTPTS,"
            f"fps={args.fps},"
            f"scale={args.panel_width}:-2:flags=lanczos,"
            f"{drawtext_filter(label)}"
            f"[v{idx}]"
        )

    chains.append("[v0][v1][v2]hstack=inputs=3,split[gif_src][gif_palette]")
    chains.append("[gif_palette]palettegen=stats_mode=diff[palette]")
    chains.append("[gif_src][palette]paletteuse=dither=sierra2_4a")
    return ";".join(chains)


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig-"))

    ffmpeg_bin = require_tool("ffmpeg")
    ffprobe_bin = require_tool("ffprobe")

    input_video = require_file(OUTPUTS_ROOT / "vipe" / "rgb" / f"{args.clip}.mp4", "Input video")
    wilor_video = require_file(OUTPUTS_ROOT / "wilor" / "videos" / f"{args.clip}.mp4", "WiLoR video")
    wilor_meshes = require_file(OUTPUTS_ROOT / "wilor" / args.clip / "meshes", "WiLoR meshes root")

    durations = [
        probe_duration(ffprobe_bin, input_video),
        probe_duration(ffprobe_bin, wilor_video),
    ]
    max_duration = min(durations)
    if args.start < 0:
        raise ValueError("--start must be non-negative")
    if args.duration <= 0:
        raise ValueError("--duration must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.panel_width <= 0:
        raise ValueError("--panel-width must be positive")
    if args.start >= max_duration:
        raise ValueError(
            f"--start ({args.start}) must be smaller than the shortest source duration ({max_duration:.3f}s)"
        )

    args.duration = min(args.duration, max_duration - args.start)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="readme_gif_") as tmp:
        temp_dir = Path(tmp)
        analysis_video = _make_grounded_wilor_video(ffmpeg_bin, wilor_meshes, temp_dir, args)

        command = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(input_video),
            "-i",
            str(wilor_video),
            "-i",
            str(analysis_video),
            "-filter_complex",
            build_filter(args),
            "-loop",
            "0",
            str(output_path),
        ]

        print("Running:")
        print(" ".join(shlex_quote(part) for part in command))
        subprocess.run(command, check=True)

    print(f"Created README GIF: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
