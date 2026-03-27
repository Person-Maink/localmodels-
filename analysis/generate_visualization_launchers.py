#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

import FILENAME as CONFIG
from npy_io import discover_frame_files, load_wilor_record


ANALYSIS_ROOT = Path(__file__).resolve().parent
VIS_ROOT = ANALYSIS_ROOT / "3D Visualization "
DEFAULT_OUTPUT_ROOT = Path(CONFIG.ANALYSIS_OUTPUT_DIR)
OUTPUTS_ROOT = Path(CONFIG.OUTPUTS_ROOT)
PROJECT_PYTHON = ANALYSIS_ROOT / ".venv" / "bin" / "python"

CAMERA_SCRIPT = VIS_ROOT / "Camera.py"
FREE_SCRIPT = VIS_ROOT / "Free.py"
WRIST_GROUNDING_SCRIPT = VIS_ROOT / "Wrist Grounding.py"
BOUNDING_BOXES_SCRIPT = VIS_ROOT / "Bounding Boxes.py"

FAMILY_NAMES = ("wilor", "hamba", "dynhamr", "vipe", "mediapipe")


def _quote(value):
    return repr(str(value))


def _shell_launcher(wrapper_name):
    return "\n".join(
        [
            "#!/usr/bin/env sh",
            'SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"',
            f'exec "{PROJECT_PYTHON}" "$SCRIPT_DIR/{wrapper_name}"',
            "",
        ]
    )


def _discover_model_clips(family):
    family_root = OUTPUTS_ROOT / family
    clips = {}
    if not family_root.exists():
        return clips

    for meshes_dir in sorted(family_root.glob("*/meshes"), key=lambda path: path.parent.name):
        if not meshes_dir.is_dir():
            continue
        clips[meshes_dir.parent.name] = meshes_dir.resolve()
    return clips


def _discover_vipe_clips():
    pose_root = OUTPUTS_ROOT / "vipe" / "pose"
    clips = {}
    if not pose_root.exists():
        return clips

    for pose_file in sorted(pose_root.glob("*.npz"), key=lambda path: path.name):
        if pose_file.is_file():
            clips[pose_file.stem] = pose_file.resolve()
    return clips


def _discover_mediapipe_clips():
    keypoints_root = OUTPUTS_ROOT / "mediapipe" / "keypoints"
    clips = {}
    if not keypoints_root.exists():
        return clips

    suffix = "_keypoints.csv"
    for csv_path in sorted(keypoints_root.glob(f"*{suffix}"), key=lambda path: path.name):
        if not csv_path.is_file() or csv_path.name.startswith(".~lock."):
            continue
        clips[csv_path.name[: -len(suffix)]] = csv_path.resolve()
    return clips


def _has_bbox_metadata(frames_root):
    try:
        discovered = discover_frame_files(frames_root, frame_dirs_glob="frame_*", file_glob="*.npy")
    except FileNotFoundError:
        return False

    for _, file_path in discovered:
        try:
            record = load_wilor_record(file_path)
        except Exception:
            continue
        if record.get("box_center") is not None and record.get("box_size") is not None:
            return True
    return False


def _camera_cli_wrapper(frames_root=None, vipe_pose_file=None, camera_poses_file=None):
    args = [repr(str(CAMERA_SCRIPT))]
    if frames_root is not None:
        args.extend(["'--frames_root'", _quote(frames_root)])
    else:
        args.extend(["'--frames_root'", "'None'"])

    if vipe_pose_file is not None:
        args.extend(["'--vipe_pose_file'", _quote(vipe_pose_file)])
    else:
        args.extend(["'--vipe_pose_file'", "'None'"])

    if camera_poses_file is not None:
        args.extend(["'--camera_poses_file'", _quote(camera_poses_file)])
    else:
        args.extend(["'--camera_poses_file'", "'None'"])

    return "\n".join(
        [
            "#!/usr/bin/env python3",
            "import subprocess",
            "import sys",
            "",
            f"ARGS = [sys.executable, {', '.join(args)}]",
            "raise SystemExit(subprocess.call(ARGS))",
            "",
        ]
    )


def _bbox_cli_wrapper(frames_root):
    return "\n".join(
        [
            "#!/usr/bin/env python3",
            "import subprocess",
            "import sys",
            "",
            f"ARGS = [sys.executable, {_quote(BOUNDING_BOXES_SCRIPT)}, '--frames_root', {_quote(frames_root)}]",
            "raise SystemExit(subprocess.call(ARGS))",
            "",
        ]
    )


def _config_injection_wrapper(target_script, config_attr, source_path):
    module_name = target_script.stem.lower().replace(" ", "_") + "_launcher_target"
    return "\n".join(
        [
            "#!/usr/bin/env python3",
            "import importlib.util",
            "import sys",
            "",
            f"ANALYSIS_ROOT = {_quote(ANALYSIS_ROOT)}",
            f"VIS_ROOT = {_quote(VIS_ROOT)}",
            "for path in (VIS_ROOT, ANALYSIS_ROOT):",
            "    if path not in sys.path:",
            "        sys.path.insert(0, path)",
            "",
            "import FILENAME as CONFIG",
            f"CONFIG.{config_attr} = {_quote(source_path)}",
            "",
            f"spec = importlib.util.spec_from_file_location('{module_name}', {_quote(target_script)})",
            "module = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(module)",
            "raise SystemExit(module.main())",
            "",
        ]
    )


def _build_family_launchers():
    launchers = {family: {} for family in FAMILY_NAMES}

    for clip_name, meshes_root in _discover_model_clips("wilor").items():
        launchers["wilor"][clip_name] = {
            "camera": _camera_cli_wrapper(frames_root=meshes_root, vipe_pose_file=None),
            "free": _config_injection_wrapper(FREE_SCRIPT, "FREE_SOURCE", meshes_root),
            "wrist_grounding": _config_injection_wrapper(WRIST_GROUNDING_SCRIPT, "WRIST_GROUNDING_SOURCE", meshes_root),
        }
        if _has_bbox_metadata(meshes_root):
            launchers["wilor"][clip_name]["bounding_boxes"] = _bbox_cli_wrapper(meshes_root)

    for clip_name, meshes_root in _discover_model_clips("hamba").items():
        launchers["hamba"][clip_name] = {
            "camera": _camera_cli_wrapper(frames_root=meshes_root, vipe_pose_file=None),
            "free": _config_injection_wrapper(FREE_SCRIPT, "FREE_SOURCE", meshes_root),
            "wrist_grounding": _config_injection_wrapper(WRIST_GROUNDING_SCRIPT, "WRIST_GROUNDING_SOURCE", meshes_root),
        }
        if _has_bbox_metadata(meshes_root):
            launchers["hamba"][clip_name]["bounding_boxes"] = _bbox_cli_wrapper(meshes_root)

    for clip_name, meshes_root in _discover_model_clips("dynhamr").items():
        camera_poses_file = meshes_root.parent / "camera_poses.npz"
        launchers["dynhamr"][clip_name] = {
            "camera": _camera_cli_wrapper(
                frames_root=None,
                vipe_pose_file=None,
                camera_poses_file=camera_poses_file if camera_poses_file.is_file() else None,
            ),
            "free": _config_injection_wrapper(FREE_SCRIPT, "FREE_SOURCE", meshes_root),
            "wrist_grounding": _config_injection_wrapper(WRIST_GROUNDING_SCRIPT, "WRIST_GROUNDING_SOURCE", meshes_root),
        }

    for clip_name, pose_file in _discover_vipe_clips().items():
        launchers["vipe"][clip_name] = {
            "camera": _camera_cli_wrapper(frames_root=None, vipe_pose_file=pose_file),
        }

    for clip_name, csv_path in _discover_mediapipe_clips().items():
        launchers["mediapipe"][clip_name] = {
            "free": _config_injection_wrapper(FREE_SCRIPT, "FREE_SOURCE", csv_path),
            "wrist_grounding": _config_injection_wrapper(WRIST_GROUNDING_SCRIPT, "WRIST_GROUNDING_SOURCE", csv_path),
        }

    return launchers


def _write_executable(path, content):
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _render_clip_files(clip_dir, desired_files):
    rendered = {}
    for name, content in desired_files.items():
        wrapper_path = clip_dir / f".{name}.py"
        rendered[wrapper_path.name] = content
        rendered[name] = _shell_launcher(wrapper_path.name)
    return rendered


def _sync_clip_dir(clip_dir, desired_files):
    clip_dir.mkdir(parents=True, exist_ok=True)
    rendered_files = _render_clip_files(clip_dir, desired_files)

    existing_names = {path.name for path in clip_dir.iterdir()}
    desired_names = set(rendered_files)

    for stale_name in sorted(existing_names - desired_names):
        stale_path = clip_dir / stale_name
        if stale_path.is_dir():
            shutil.rmtree(stale_path)
        else:
            stale_path.unlink()

    for name, content in rendered_files.items():
        _write_executable(clip_dir / name, content)


def _sync_family_dir(family_dir, desired_clips):
    family_dir.mkdir(parents=True, exist_ok=True)

    existing_names = {path.name for path in family_dir.iterdir()}
    desired_names = set(desired_clips)

    for stale_name in sorted(existing_names - desired_names):
        stale_path = family_dir / stale_name
        if stale_path.is_dir():
            shutil.rmtree(stale_path)
        else:
            stale_path.unlink()

    for clip_name, files in desired_clips.items():
        _sync_clip_dir(family_dir / clip_name, files)


def generate_launcher_tree(output_root):
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    desired_tree = _build_family_launchers()

    summary = {}
    for family in FAMILY_NAMES:
        family_dir = output_root / family
        desired_clips = desired_tree.get(family, {})
        _sync_family_dir(family_dir, desired_clips)
        summary[family] = {
            "clips": len(desired_clips),
            "launchers": sum(len(files) for files in desired_clips.values()),
        }

    return output_root, summary


def main():
    parser = argparse.ArgumentParser(description="Generate per-clip visualization launcher scripts under ANALYSIS_OUTPUT_DIR.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for generated visualization launchers.",
    )
    args = parser.parse_args()

    output_root, summary = generate_launcher_tree(args.output_root)
    print(f"Generated launcher tree under: {output_root}")
    for family in FAMILY_NAMES:
        info = summary[family]
        print(f"  {family}: clips={info['clips']}, launchers={info['launchers']}")


if __name__ == "__main__":
    main()
