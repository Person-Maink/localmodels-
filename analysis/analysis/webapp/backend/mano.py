from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict

import numpy as np

from mano_pickle import load_mano_pickle


def _install_numpy_legacy_aliases() -> None:
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


@lru_cache(maxsize=2)
def load_mano_assets(mano_right_path: str) -> Dict[str, np.ndarray]:
    path = Path(mano_right_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MANO_RIGHT.pkl not found: {path}")

    _install_numpy_legacy_aliases()
    mano = load_mano_pickle(path)

    j_regressor = mano["J_regressor"]
    if hasattr(j_regressor, "toarray"):
        j_regressor = j_regressor.toarray()

    faces = _to_numpy_array(mano["f"], dtype=np.int32)
    shapedirs = _to_numpy_array(mano["shapedirs"], dtype=np.float32)
    posedirs = _to_numpy_array(mano["posedirs"], dtype=np.float32)
    weights = _to_numpy_array(mano["weights"], dtype=np.float32)
    v_template = _to_numpy_array(mano["v_template"], dtype=np.float32)
    kintree_table = _to_numpy_array(mano["kintree_table"], dtype=np.int64)
    j_regressor = _to_numpy_array(j_regressor, dtype=np.float32)

    return {
        "path": str(path),
        "faces_right": faces,
        "faces_left": faces[:, [0, 2, 1]].copy(),
        "j_regressor": j_regressor,
        "shapedirs": shapedirs,
        "posedirs": posedirs,
        "weights": weights,
        "v_template": v_template,
        "parents": _build_parents(kintree_table),
        "num_betas": int(shapedirs.shape[2]),
    }


def axis_angle_to_matrix(axis_angles):
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


def coerce_rotations(value, joint_count: int, field_name: str, source_path: Path) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape == (joint_count, 3, 3):
        return arr
    if arr.shape == (1, joint_count, 3, 3):
        return arr[0]
    if joint_count == 1 and arr.shape == (3, 3):
        return arr.reshape(1, 3, 3)
    if arr.shape == (joint_count, 3):
        return axis_angle_to_matrix(arr)
    if arr.shape == (1, joint_count, 3):
        return axis_angle_to_matrix(arr[0])
    if joint_count == 1 and arr.shape in {(3,), (1, 3)}:
        return axis_angle_to_matrix(arr)[0:1]
    if arr.shape == (joint_count * 3,):
        return axis_angle_to_matrix(arr.reshape(joint_count, 3))
    if arr.shape == (1, joint_count * 3):
        return axis_angle_to_matrix(arr.reshape(joint_count, 3))
    raise ValueError(f"Unsupported {field_name} shape in {source_path}: {arr.shape}")


def transform_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.zeros((4, 4), dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    transform[3, 3] = 1.0
    return transform


def mano_forward(mano, betas, global_orient, hand_pose, right):
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
                transforms[batch_idx, joint_idx] = transform_matrix(full_pose[batch_idx, joint_idx], joint)
            else:
                joint_rel = joint - joints[batch_idx, parent_idx]
                transforms[batch_idx, joint_idx] = (
                    transforms[batch_idx, parent_idx]
                    @ transform_matrix(full_pose[batch_idx, joint_idx], joint_rel)
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
