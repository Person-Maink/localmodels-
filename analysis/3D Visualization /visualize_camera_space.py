import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
WILOR_ROOT_DIR = REPO_ROOT / "models" / "wilor_hands"
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))
if str(WILOR_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(WILOR_ROOT_DIR))

from mano_pickle import load_mano_pickle


LEFT_COLOR = "royalblue"
RIGHT_COLOR = "crimson"
MESH_ALPHA = 0.60
_MISSING_GLOBAL_ORIENT_WARNED = False


def _apply_numpy_legacy_aliases():
    alias_map = {
        "bool": np.bool_,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
        "nan": float("nan"),
        "inf": float("inf"),
    }
    for name, value in alias_map.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def _load_record(path: Path):
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.dtype == object:
        return payload.item()
    if hasattr(payload, "item"):
        return payload.item()
    raise ValueError(f"Unsupported WiLoR record format: {path}")


def _axis_angle_to_matrix(axis_angle):
    axis_angle = np.asarray(axis_angle, dtype=np.float32).reshape(3)
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = axis_angle / angle
    x, y, z = axis
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float32,
    )


def _normalize_global_orient(record):
    global _MISSING_GLOBAL_ORIENT_WARNED
    pred_mano_params = record.get("pred_mano_params", {})
    if "global_orient" not in pred_mano_params:
        if not _MISSING_GLOBAL_ORIENT_WARNED:
            print(
                "Warning: some WiLoR records do not contain pred_mano_params/global_orient; "
                "skipping global-orient removal for those files."
            )
            _MISSING_GLOBAL_ORIENT_WARNED = True
        return np.eye(3, dtype=np.float32)
    global_orient = np.asarray(pred_mano_params.get("global_orient"), dtype=np.float32)
    if global_orient.shape == (1, 3, 3):
        return global_orient[0]
    if global_orient.shape == (3, 3):
        return global_orient
    if global_orient.shape == (1, 3):
        return _axis_angle_to_matrix(global_orient[0])
    if global_orient.shape == (3,):
        return _axis_angle_to_matrix(global_orient)
    raise ValueError(
        f"Unsupported global_orient shape in {record.get('_path', '<memory>')}: {global_orient.shape}"
    )


def _frame_records(mesh_root: Path):
    records = {}
    for path in sorted(mesh_root.glob("frame_*/*.npy")):
        record = _load_record(path)
        frame_id = int(record.get("frame_id", int(path.parent.name.split("_")[-1])))
        record["_path"] = path
        records.setdefault(frame_id, []).append(record)
    return records


def _resolve_video_dir(wilor_root: Path, video_name: str) -> Path:
    video_dir = wilor_root / video_name
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Could not find WiLoR output directory: {video_dir}")
    mesh_dir = video_dir / "meshes"
    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"Could not find WiLoR mesh cache: {mesh_dir}")
    return video_dir


def _load_mano_faces(mano_model_path: str):
    if not mano_model_path:
        raise FileNotFoundError("A MANO model path is required for mesh visualization.")

    candidate = Path(mano_model_path)
    candidates = []
    if candidate.is_dir():
        candidates.extend(
            [
                candidate / "MANO_RIGHT.pkl",
                candidate / "mano_data" / "MANO_RIGHT.pkl",
            ]
        )
    else:
        candidates.append(candidate)

    for path in candidates:
        if not path.is_file():
            continue
        _apply_numpy_legacy_aliases()
        payload = load_mano_pickle(path)
        faces = payload.get("f")
        if faces is not None:
            return np.asarray(faces, dtype=np.int32)

    raise FileNotFoundError(
        f"Could not find MANO_RIGHT.pkl under {mano_model_path}. "
        "Pass --mano_model_path pointing to the MANO asset directory or file."
    )


def _camera_space_record(record):
    cam_t = np.asarray(record["cam_t"], dtype=np.float32)
    global_orient = _normalize_global_orient(record)
    verts_cam = np.asarray(record["verts"], dtype=np.float32) + cam_t[None, :]
    joints_cam = np.asarray(record["joints"], dtype=np.float32) + cam_t[None, :]
    wrist = joints_cam[0]
    inv_global_orient = global_orient.T
    verts = (verts_cam - wrist[None, :]) @ inv_global_orient + wrist[None, :]
    joints = (joints_cam - wrist[None, :]) @ inv_global_orient + wrist[None, :]
    frame_id = record.get("frame_id")
    if frame_id is None:
        frame_id = int(record["_path"].parent.name.split("_")[-1])
    return {
        "frame_id": int(frame_id),
        "right": int(round(float(record["right"]))),
        "verts": verts,
        "joints": joints,
        "score": float(record.get("detection_confidence") or 1.0),
    }


def _collect_frames(mesh_dir: Path):
    records_by_frame = _frame_records(mesh_dir)
    frames = {}
    for frame_id, records in sorted(records_by_frame.items()):
        frames[frame_id] = [_camera_space_record(record) for record in records]
    if not frames:
        raise RuntimeError(f"No WiLoR records found under {mesh_dir}")
    return [(frame_id, frames[frame_id]) for frame_id in sorted(frames)]


def _compute_scene_center(frames):
    all_points = []
    for _, frame_records in frames:
        for record in frame_records:
            all_points.append(record["verts"])
    stacked = np.concatenate(all_points, axis=0)
    return stacked.mean(axis=0)


def _color_for_hand(is_right: int) -> str:
    return RIGHT_COLOR if int(is_right) == 1 else LEFT_COLOR


def _make_frame_actor(frame_records, faces, vedo):
    actors = []
    for record in sorted(frame_records, key=lambda item: item["score"]):
        mesh = vedo.Mesh([record["verts"], faces])
        mesh.c(_color_for_hand(record["right"]))
        mesh.alpha(MESH_ALPHA)
        mesh.lighting("default")
        actors.append(mesh)

    if not actors:
        return None
    if len(actors) == 1:
        return actors[0]
    return sum(actors[1:], actors[0])


def _make_origin_marker(scene_center, vedo):
    return vedo.Point(scene_center, r=18, c="black")


def run_vedo_viewer(
    wilor_root: str,
    video_name: str,
    mano_model_path: str,
):
    try:
        import vedo
        from vedo.applications import AnimationPlayer
    except ImportError as exc:
        raise ImportError(
            "This viewer needs vedo. Install it in your environment first, for example with "
            "`pip install vedo vtk` or your project package manager."
        ) from exc

    wilor_root_path = Path(wilor_root).resolve()
    video_dir = _resolve_video_dir(wilor_root_path, video_name)
    mesh_dir = video_dir / "meshes"
    frames = _collect_frames(mesh_dir)
    faces = _load_mano_faces(mano_model_path)
    scene_center = _compute_scene_center(frames)

    print(f"Using WiLoR output: {video_dir}")
    print(f"Loaded {len(frames)} frame(s)")
    print("Displaying hands in camera coordinates via verts_cam = verts + cam_t")
    print("Global orientation is removed around the wrist for each frame.")
    print("Note: this is not true world space without external camera extrinsics.")

    actor = _make_frame_actor(frames[0][1], faces, vedo)
    origin_marker = _make_origin_marker(scene_center, vedo)

    def update_scene(i):
        nonlocal actor
        if actor is not None:
            plt.remove(actor)

        frame_id, frame_records = frames[i]
        actor = _make_frame_actor(frame_records, faces, vedo)
        if actor is not None:
            plt.add(actor)

        if title_actor is not None:
            title_actor.text(f"WiLoR camera-space meshes, global orient removed | frame {frame_id:06d}")

        plt.render()

    plt = AnimationPlayer(update_scene, irange=[0, len(frames) - 1])
    if actor is not None:
        plt += actor
    plt += origin_marker

    frame_id0 = frames[0][0]
    title_actor = vedo.Text2D(
        f"WiLoR camera-space meshes, global orient removed | frame {frame_id0:06d}",
        pos="top-middle",
        c="black",
        s=0.9,
    )
    note_actor = vedo.Text2D(
        "camera-space only, not world-space",
        pos="bottom-left",
        c="dimgray",
        s=0.7,
    )
    plt += title_actor
    plt += note_actor

    plt.show(bg="white", axes=1, viewup="z")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Interactive vedo viewer for saved WiLoR meshes. "
            "It reconstructs per-frame camera-space meshes using verts_cam = verts + cam_t. "
            "Because WiLoR alone does not estimate a shared scene frame, this is camera-space, not true world-space."
        )
    )
    parser.add_argument(
        "--wilor_root",
        type=str,
        default=str(REPO_ROOT / "outputs" / "wilor"),
        help="Root directory containing per-video WiLoR outputs.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="me 1",
        help="Video stem to visualize, matching the WiLoR output subdirectory name.",
    )
    parser.add_argument(
        "--mano_model_path",
        type=str,
        default=str(WILOR_ROOT_DIR / "mano_data"),
        help="Path to MANO assets or directly to MANO_RIGHT.pkl.",
    )
    args = parser.parse_args()

    run_vedo_viewer(
        wilor_root=args.wilor_root,
        video_name=args.video,
        mano_model_path=args.mano_model_path,
    )


if __name__ == "__main__":
    main()
