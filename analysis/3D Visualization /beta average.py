import argparse
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from mano_pickle import load_mano_pickle
from npy_io import discover_frame_files, resolve_model_record_root

try:
    from FILENAME import WRIST_JOINT_IDX as DEFAULT_WRIST_JOINT_IDX
except Exception:
    DEFAULT_WRIST_JOINT_IDX = 0


LEFT_COLOR = "royalblue"
RIGHT_COLOR = "crimson"
MESH_ALPHA = 0.60


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
        if name not in np.__dict__:
            setattr(np, name, value)


def _to_numpy_array(value, dtype=np.float32):
    if hasattr(value, "r"):
        value = value.r
    return np.asarray(value, dtype=dtype)


def _load_record(path: Path):
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.dtype == object and payload.shape == ():
        return payload.item()
    if hasattr(payload, "item"):
        return payload.item()
    raise ValueError(f"Unsupported WiLoR record format: {path}")


def _normalize_video_name(video_name: str) -> str:
    name = Path(str(video_name)).name
    if name.lower().endswith(".mp4"):
        return Path(name).stem
    return name


def _resolve_video_dir(wilor_root: Path, video_name: str) -> Path:
    video_dir = wilor_root / _normalize_video_name(video_name)
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Could not find WiLoR output directory: {video_dir}")
    mesh_dir = video_dir / "meshes"
    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"Could not find WiLoR mesh cache: {mesh_dir}")
    return video_dir


def _resolve_model_source_path(source_path) -> tuple[Path, Path]:
    resolved_source = Path(source_path).expanduser().resolve()
    record_root = resolve_model_record_root(resolved_source)
    if record_root is None:
        raise FileNotFoundError(
            "Could not find a compatible saved model record root under "
            f"{resolved_source}. Expected raw frame records in the source path or its 'meshes' child."
        )
    return resolved_source, record_root


def _build_parents(kintree_table: np.ndarray) -> np.ndarray:
    if kintree_table.ndim != 2 or kintree_table.shape[0] != 2:
        raise ValueError(f"Invalid MANO kintree_table shape: {kintree_table.shape}")

    parents = np.full((kintree_table.shape[1],), -1, dtype=np.int64)
    node_to_col = {int(kintree_table[1, idx]): idx for idx in range(kintree_table.shape[1])}
    for idx in range(1, kintree_table.shape[1]):
        parent_node = int(kintree_table[0, idx])
        if parent_node not in node_to_col:
            raise ValueError(f"Could not resolve parent node {parent_node} in MANO kintree_table")
        parents[idx] = node_to_col[parent_node]
    return parents


def _load_mano_model(mano_model_path: str):
    if not mano_model_path:
        raise FileNotFoundError("A MANO model path is required for averaged-beta visualization.")

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
        mano = load_mano_pickle(path)

        v_template = _to_numpy_array(mano["v_template"], dtype=np.float32)
        shapedirs = _to_numpy_array(mano["shapedirs"], dtype=np.float32)
        posedirs = _to_numpy_array(mano["posedirs"], dtype=np.float32)
        weights = _to_numpy_array(mano["weights"], dtype=np.float32)
        faces = _to_numpy_array(mano["f"], dtype=np.int32)
        kintree_table = _to_numpy_array(mano["kintree_table"], dtype=np.int64)
        j_regressor = mano["J_regressor"]
        if hasattr(j_regressor, "toarray"):
            j_regressor = j_regressor.toarray()
        j_regressor = _to_numpy_array(j_regressor, dtype=np.float32)

        if v_template.shape != (778, 3):
            raise ValueError(f"Unexpected MANO v_template shape in {path}: {v_template.shape}")
        if shapedirs.ndim != 3 or shapedirs.shape[:2] != (778, 3):
            raise ValueError(f"Unexpected MANO shapedirs shape in {path}: {shapedirs.shape}")
        if posedirs.shape != (778, 3, 135):
            raise ValueError(f"Unexpected MANO posedirs shape in {path}: {posedirs.shape}")
        if weights.shape != (778, 16):
            raise ValueError(f"Unexpected MANO weights shape in {path}: {weights.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"Unexpected MANO face topology in {path}: {faces.shape}")
        if j_regressor.shape != (16, 778):
            raise ValueError(f"Unexpected MANO joint regressor shape in {path}: {j_regressor.shape}")

        return {
            "path": path,
            "v_template": v_template,
            "shapedirs": shapedirs,
            "posedirs": posedirs,
            "weights": weights,
            "faces_right": faces,
            "faces_left": faces[:, [0, 2, 1]].copy(),
            "j_regressor": j_regressor,
            "parents": _build_parents(kintree_table),
            "num_betas": int(shapedirs.shape[2]),
        }

    raise FileNotFoundError(
        f"Could not find MANO_RIGHT.pkl under {mano_model_path}. "
        "Pass --mano_model_path pointing to the MANO asset directory or file."
    )


def _axis_angle_to_matrix(axis_angles):
    axis_angles = np.asarray(axis_angles, dtype=np.float32).reshape(-1, 3)
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None], axis_angles.shape[0], axis=0)

    for idx, axis_angle in enumerate(axis_angles):
        angle = float(np.linalg.norm(axis_angle))
        if angle < 1e-8:
            continue

        axis = axis_angle / angle
        x, y, z = axis
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        one_c = 1.0 - c
        rotations[idx] = np.asarray(
            [
                [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
                [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
                [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
            ],
            dtype=np.float32,
        )

    return rotations


def _coerce_rotations(value, joint_count: int, field_name: str, source_path: Path) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)

    if arr.shape == (joint_count, 3, 3):
        return arr
    if arr.shape == (1, joint_count, 3, 3):
        return arr[0]
    if joint_count == 1 and arr.shape == (3, 3):
        return arr.reshape(1, 3, 3)
    if arr.shape == (joint_count, 3):
        return _axis_angle_to_matrix(arr)
    if arr.shape == (1, joint_count, 3):
        return _axis_angle_to_matrix(arr[0])
    if joint_count == 1 and arr.shape == (3,):
        return _axis_angle_to_matrix(arr)[0:1]
    if joint_count == 1 and arr.shape == (1, 3):
        return _axis_angle_to_matrix(arr)[0:1]
    if arr.shape == (joint_count * 3,):
        return _axis_angle_to_matrix(arr.reshape(joint_count, 3))
    if arr.shape == (1, joint_count * 3):
        return _axis_angle_to_matrix(arr.reshape(joint_count, 3))

    raise ValueError(
        f"Unsupported {field_name} shape in {source_path}: {arr.shape}"
    )


def _extract_frame_id(record: dict, path: Path) -> int:
    frame_id = record.get("frame_id")
    if frame_id is not None:
        return int(np.asarray(frame_id, dtype=np.int32).reshape(()))
    return int(path.parent.name.split("_")[-1])


def _extract_required_record(path: Path, num_betas: int):
    record = _load_record(path)
    if not isinstance(record, dict):
        raise ValueError(f"Expected a dict payload in {path}")

    pred_mano_params = record.get("pred_mano_params")
    if not isinstance(pred_mano_params, dict):
        raise RuntimeError(f"Missing pred_mano_params in {path}")

    if "betas" not in pred_mano_params:
        raise RuntimeError(f"Missing pred_mano_params/betas in {path}")
    if "global_orient" not in pred_mano_params:
        raise RuntimeError(f"Missing pred_mano_params/global_orient in {path}")
    if "hand_pose" not in pred_mano_params:
        raise RuntimeError(f"Missing pred_mano_params/hand_pose in {path}")
    if "cam_t" not in record:
        raise RuntimeError(f"Missing cam_t in {path}")
    if "right" not in record:
        raise RuntimeError(f"Missing right handedness flag in {path}")

    betas = np.asarray(pred_mano_params["betas"], dtype=np.float32).reshape(-1)
    if betas.shape != (num_betas,):
        raise ValueError(f"Unexpected betas shape in {path}: {betas.shape}, expected ({num_betas},)")
    if not np.all(np.isfinite(betas)):
        raise ValueError(f"Non-finite betas found in {path}")

    global_orient = _coerce_rotations(
        pred_mano_params["global_orient"],
        joint_count=1,
        field_name="global_orient",
        source_path=path,
    )
    hand_pose = _coerce_rotations(
        pred_mano_params["hand_pose"],
        joint_count=15,
        field_name="hand_pose",
        source_path=path,
    )

    cam_t = np.asarray(record["cam_t"], dtype=np.float32).reshape(3)
    if not np.all(np.isfinite(cam_t)):
        raise ValueError(f"Non-finite cam_t found in {path}")

    right = int(round(float(np.asarray(record["right"]).reshape(()))))
    if right not in {0, 1}:
        raise ValueError(f"Unsupported handedness value in {path}: {right}")

    score_value = record.get("detection_confidence", 1.0)
    if score_value is None:
        score = 1.0
    else:
        score = float(score_value)

    return {
        "path": path,
        "frame_id": _extract_frame_id(record, path),
        "betas": betas,
        "global_orient": global_orient,
        "hand_pose": hand_pose,
        "cam_t": cam_t,
        "right": right,
        "score": score,
    }


def _transform_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.zeros((4, 4), dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    transform[3, 3] = 1.0
    return transform


def _mano_forward(mano, betas, global_orient, hand_pose, right):
    betas = np.asarray(betas, dtype=np.float32)
    global_orient = np.asarray(global_orient, dtype=np.float32)
    hand_pose = np.asarray(hand_pose, dtype=np.float32)
    right = np.asarray(right, dtype=np.int32).reshape(-1)

    batch_size = betas.shape[0]
    full_pose = np.concatenate([global_orient, hand_pose], axis=1)

    v_shaped = mano["v_template"][None] + np.einsum("bl,vcl->bvc", betas, mano["shapedirs"])
    joints = np.einsum("jv,bvc->bjc", mano["j_regressor"], v_shaped)

    ident = np.eye(3, dtype=np.float32)
    pose_feature = (full_pose[:, 1:] - ident).reshape(batch_size, -1)
    pose_offsets = np.einsum("bp,vcp->bvc", pose_feature, mano["posedirs"])
    v_posed = v_shaped + pose_offsets

    transforms = np.zeros((batch_size, 16, 4, 4), dtype=np.float32)
    for batch_idx in range(batch_size):
        for joint_idx, parent_idx in enumerate(mano["parents"]):
            joint = joints[batch_idx, joint_idx]
            if parent_idx == -1:
                transforms[batch_idx, joint_idx] = _transform_matrix(full_pose[batch_idx, joint_idx], joint)
            else:
                joint_rel = joint - joints[batch_idx, parent_idx]
                transforms[batch_idx, joint_idx] = (
                    transforms[batch_idx, parent_idx]
                    @ _transform_matrix(full_pose[batch_idx, joint_idx], joint_rel)
                )

    rel_transforms = transforms.copy()
    for batch_idx in range(batch_size):
        for joint_idx in range(16):
            rel_transforms[batch_idx, joint_idx, :3, 3] -= (
                transforms[batch_idx, joint_idx, :3, :3] @ joints[batch_idx, joint_idx]
            )

    skinning = np.einsum("vj,bjkl->bvkl", mano["weights"], rel_transforms)
    v_homo = np.concatenate(
        [v_posed, np.ones((batch_size, v_posed.shape[1], 1), dtype=np.float32)],
        axis=2,
    )
    verts = np.einsum("bvkl,bvl->bvk", skinning, v_homo)[:, :, :3]

    mirror = (2 * right - 1).astype(np.float32).reshape(-1, 1)
    verts[:, :, 0] *= mirror
    return verts


def _wrist_ground_vertices(verts: np.ndarray, mano, wrist_joint_idx: int) -> np.ndarray:
    verts = np.asarray(verts, dtype=np.float32)
    wrist_joint_idx = int(wrist_joint_idx)
    if wrist_joint_idx < 0 or wrist_joint_idx >= mano["j_regressor"].shape[0]:
        raise ValueError(
            f"Invalid wrist_joint_idx={wrist_joint_idx}; expected [0, {mano['j_regressor'].shape[0] - 1}]"
        )

    wrist = mano["j_regressor"][wrist_joint_idx] @ verts
    return verts - wrist.reshape(1, 3)


def load_average_beta_frames(
    wilor_root: str,
    video_name: str,
    mano_model_path: str,
    wrist_ground: bool = False,
    wrist_joint_idx: int = DEFAULT_WRIST_JOINT_IDX,
):
    wilor_root_path = Path(wilor_root).resolve()
    video_dir = _resolve_video_dir(wilor_root_path, video_name)
    bundle = load_average_beta_frames_for_source(
        source_path=video_dir,
        mano_model_path=mano_model_path,
        wrist_ground=wrist_ground,
        wrist_joint_idx=wrist_joint_idx,
    )
    bundle["video_dir"] = video_dir
    return bundle


def load_average_beta_frames_for_source(
    source_path,
    mano_model_path: str,
    wrist_ground: bool = False,
    wrist_joint_idx: int = DEFAULT_WRIST_JOINT_IDX,
):
    resolved_source, record_root = _resolve_model_source_path(source_path)
    mano = _load_mano_model(mano_model_path)

    records = []
    for _, path in discover_frame_files(record_root, frame_dirs_glob="frame_*", file_glob="*.npy"):
        records.append(_extract_required_record(path, num_betas=mano["num_betas"]))

    if not records:
        raise RuntimeError(f"No compatible model records found under {record_root}")

    average_betas = np.stack([record["betas"] for record in records], axis=0).mean(axis=0)
    betas_batch = np.broadcast_to(average_betas[None], (len(records), average_betas.shape[0])).copy()
    global_orient = np.stack([record["global_orient"] for record in records], axis=0)
    hand_pose = np.stack([record["hand_pose"] for record in records], axis=0)
    right = np.asarray([record["right"] for record in records], dtype=np.int32)
    cam_t = np.stack([record["cam_t"] for record in records], axis=0)

    verts_local = _mano_forward(
        mano=mano,
        betas=betas_batch,
        global_orient=global_orient,
        hand_pose=hand_pose,
        right=right,
    )
    verts_camera = verts_local + cam_t[:, None, :]

    frames = {}
    for record, verts in zip(records, verts_camera):
        if wrist_ground:
            verts = _wrist_ground_vertices(verts, mano, wrist_joint_idx=wrist_joint_idx)
        frames.setdefault(record["frame_id"], []).append(
            {
                "verts": verts,
                "right": record["right"],
                "score": record["score"],
            }
        )

    ordered_frames = [(frame_id, frames[frame_id]) for frame_id in sorted(frames)]
    return {
        "source_path": resolved_source,
        "record_root": record_root,
        "mano": mano,
        "frames": ordered_frames,
        "average_betas": average_betas,
        "record_count": len(records),
        "wrist_ground": bool(wrist_ground),
        "wrist_joint_idx": int(wrist_joint_idx),
    }


def _color_for_hand(is_right: int) -> str:
    return RIGHT_COLOR if int(is_right) == 1 else LEFT_COLOR


def _make_frame_actor(frame_records, mano, vedo):
    actors = []
    for record in sorted(frame_records, key=lambda item: item["score"]):
        faces = mano["faces_right"] if int(record["right"]) == 1 else mano["faces_left"]
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


def run_vedo_viewer(
    wilor_root: str,
    video_name: str,
    mano_model_path: str,
    wrist_ground: bool = False,
    wrist_joint_idx: int = DEFAULT_WRIST_JOINT_IDX,
):
    try:
        import vedo
        from vedo.applications import AnimationPlayer
    except ImportError as exc:
        raise ImportError(
            "This viewer needs vedo. Install it in your environment first, for example with "
            "`pip install vedo vtk` or your project package manager."
        ) from exc

    bundle = load_average_beta_frames(
        wilor_root=wilor_root,
        video_name=video_name,
        mano_model_path=mano_model_path,
        wrist_ground=wrist_ground,
        wrist_joint_idx=wrist_joint_idx,
    )
    video_dir = bundle["video_dir"]
    mano = bundle["mano"]
    frames = bundle["frames"]
    average_betas = bundle["average_betas"]
    mode_label = "wrist-grounded" if bundle["wrist_ground"] else "camera-space"

    print(f"Using WiLoR output: {video_dir}")
    print(f"Using MANO file: {mano['path']}")
    print(f"Loaded {len(frames)} frame(s) and {bundle['record_count']} hand record(s)")
    if bundle["wrist_ground"]:
        print(
            "Displaying wrist-grounded meshes rebuilt with one sequence-average beta vector "
            f"using wrist joint {bundle['wrist_joint_idx']}."
        )
    else:
        print("Displaying camera-space meshes rebuilt with one sequence-average beta vector.")
    print(f"Average betas: {np.array2string(average_betas, precision=5, separator=', ')}")

    actor = _make_frame_actor(frames[0][1], mano, vedo)

    def update_scene(index: int):
        nonlocal actor
        if actor is not None:
            plt.remove(actor)

        frame_id, frame_records = frames[index]
        actor = _make_frame_actor(frame_records, mano, vedo)
        if actor is not None:
            plt.add(actor)

        if title_actor is not None:
            title_actor.text(f"WiLoR {mode_label} meshes | sequence-average betas | frame {frame_id:06d}")

        plt.render()

    plt = AnimationPlayer(update_scene, irange=[0, len(frames) - 1])
    if actor is not None:
        plt += actor

    frame_id0 = frames[0][0]
    title_actor = vedo.Text2D(
        f"WiLoR {mode_label} meshes | sequence-average betas | frame {frame_id0:06d}",
        pos="top-middle",
        c="black",
        s=0.9,
    )
    note_actor = vedo.Text2D(
        (
            "wrist-grounded; all records use the same averaged betas"
            if bundle["wrist_ground"]
            else "camera-space only; all records use the same averaged betas"
        ),
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
            "Interactive vedo viewer for saved WiLoR meshes that rebuilds every hand with a single "
            "sequence-average beta vector while keeping each frame's MANO pose and camera translation."
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
        default="me 4",
        help="Video filename or stem. '.mp4' is stripped to match the WiLoR output folder name.",
    )
    parser.add_argument(
        "--mano_model_path",
        type=str,
        default=str(ANALYSIS_ROOT / "mano_data"),
        help="Path to MANO assets or directly to MANO_RIGHT.pkl.",
    )
    parser.add_argument(
        "--wrist_ground",
        default=True,
        help="Subtract the wrist joint from each reconstructed hand before visualization.",
    )
    args = parser.parse_args()

    run_vedo_viewer(
        wilor_root=args.wilor_root,
        video_name=args.video,
        mano_model_path=args.mano_model_path,
        wrist_ground=args.wrist_ground,
    )


if __name__ == "__main__":
    main()
