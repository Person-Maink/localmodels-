import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from utils_new import render_rgba_multiple
from visualize import images_to_video
from wilor.models import MANO
from wilor.utils.geometry import perspective_projection, rot6d_to_rotmat


LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)
DEFAULT_FFT_JOINTS = (0, 4, 8, 12, 16, 20)


@dataclass
class StrideConfig:
    iters: int = 300
    lr: float = 0.05
    obs_weight: float = 10.0
    reproj_weight: float = 2.0
    shape_weight: float = 5.0
    cam_smooth_weight: float = 25.0
    pose_smooth_weight: float = 1.5
    joint_smooth_weight: float = 0.5
    anchor_weight: float = 0.5
    fft_weight: float = 0.0
    fft_band_low_hz: float | None = None
    fft_band_high_hz: float | None = None
    fps: float = 30.0
    pose_rank: int = 32
    cam_rank: int = 16


def _load_record(path: Path):
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.dtype == object:
        return payload.item()
    if hasattr(payload, "item"):
        return payload.item()
    raise ValueError(f"Unsupported WiLoR record format: {path}")


def _frame_records(mesh_root: Path):
    records = {}
    for path in sorted(mesh_root.glob("frame_*/*.npy")):
        record = _load_record(path)
        frame_id = int(record.get("frame_id", int(path.parent.name.split("_")[-1])))
        record["_path"] = path
        records.setdefault(frame_id, []).append(record)
    return records


def _parse_target_hand(target_hand):
    if target_hand in {"auto", None}:
        return "auto"
    if str(target_hand).lower() in {"right", "r", "1"}:
        return 1
    if str(target_hand).lower() in {"left", "l", "0"}:
        return 0
    raise ValueError(f"Unsupported target_hand value: {target_hand}")


def _dominant_handedness(records_by_frame):
    score_by_hand = {}
    for frame_records in records_by_frame.values():
        for record in frame_records:
            hand = int(round(float(record["right"])))
            score_by_hand[hand] = score_by_hand.get(hand, 0.0) + float(record.get("detection_confidence") or 1.0)
    if not score_by_hand:
        raise RuntimeError("No WiLoR detections found for STRIDE refinement.")
    return max(score_by_hand.items(), key=lambda item: item[1])[0]


def _pick_track(records_by_frame, target_hand="auto"):
    chosen_hand = _dominant_handedness(records_by_frame) if target_hand == "auto" else int(target_hand)
    selected = []
    prev_center = None
    for frame_id in sorted(records_by_frame):
        candidates = [
            record for record in records_by_frame[frame_id]
            if int(round(float(record["right"]))) == chosen_hand
        ]
        if not candidates:
            continue

        def score(record):
            conf = float(record.get("detection_confidence") or 1.0)
            if prev_center is None:
                return conf
            center = np.asarray(record["box_center"], dtype=np.float32)
            return conf - 0.01 * float(np.linalg.norm(center - prev_center))

        best = max(candidates, key=score)
        selected.append(best)
        prev_center = np.asarray(best["box_center"], dtype=np.float32)

    if not selected:
        raise RuntimeError("Could not select a dominant hand track for STRIDE refinement.")
    return selected


def _linear_fill(values: np.ndarray, observed_mask: np.ndarray):
    output = values.copy()
    time_index = np.arange(len(values), dtype=np.float32)
    flat = output.reshape(len(values), -1)
    for column in range(flat.shape[1]):
        observed = observed_mask & np.isfinite(flat[:, column])
        if not observed.any():
            continue
        if observed.sum() == 1:
            flat[:, column] = flat[np.where(observed)[0][0], column]
            continue
        flat[:, column] = np.interp(time_index, time_index[observed], flat[observed, column]).astype(np.float32)
    return flat.reshape(values.shape)


def _rotmat_to_rot6d(rotmats: torch.Tensor) -> torch.Tensor:
    return rotmats[..., :2].permute(0, 1, 3, 2).reshape(rotmats.shape[0], rotmats.shape[1], 6)


def _build_temporal_basis(length: int, rank: int, device: torch.device, dtype: torch.dtype):
    rank = max(1, min(rank, length))
    t = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    k = torch.arange(rank, device=device, dtype=dtype).unsqueeze(0)
    basis = torch.cos(np.pi * (t + 0.5) * k / float(length))
    basis[:, 0] = 1.0
    basis = basis / basis.norm(dim=0, keepdim=True).clamp_min(1e-6)
    return basis


def _second_difference(sequence: torch.Tensor):
    if sequence.shape[0] < 3:
        return sequence.new_zeros((0, *sequence.shape[1:]))
    return sequence[2:] - 2.0 * sequence[1:-1] + sequence[:-2]


def _stack_records(records, image_folder=None, video_name=None):
    frame_ids = np.asarray([int(record["frame_id"]) for record in records], dtype=np.int32)
    full_frame_ids = np.arange(frame_ids.min(), frame_ids.max() + 1, dtype=np.int32)
    observed_mask = np.isin(full_frame_ids, frame_ids)

    by_frame = {int(record["frame_id"]): record for record in records}
    template = records[0]

    def collect(key, nested=None):
        values = []
        for frame_id in full_frame_ids:
            record = by_frame.get(int(frame_id))
            if record is None:
                values.append(np.full_like(values[-1], np.nan) if values else np.full_like(
                    np.asarray(template[key] if nested is None else template[key][nested], dtype=np.float32),
                    np.nan,
                    dtype=np.float32,
                ))
                continue
            value = record[key] if nested is None else record[key][nested]
            values.append(np.asarray(value, dtype=np.float32))
        return _linear_fill(np.stack(values).astype(np.float32), observed_mask)

    seq = {
        "frame_ids": full_frame_ids,
        "observed_mask": observed_mask.astype(bool),
        "right": int(round(float(template["right"]))),
        "detection_confidence": _linear_fill(
            np.asarray(
                [
                    float(by_frame[int(fid)].get("detection_confidence") or 1.0) if int(fid) in by_frame else np.nan
                    for fid in full_frame_ids
                ],
                dtype=np.float32,
            )[:, None],
            observed_mask,
        )[:, 0],
        "box_center": collect("box_center"),
        "box_size": collect("box_size"),
        "cam_t": collect("cam_t"),
        "pred_keypoints_2d": collect("pred_keypoints_2d"),
        "joints": collect("joints"),
        "global_orient": collect("pred_mano_params", "global_orient"),
        "hand_pose": collect("pred_mano_params", "hand_pose"),
        "betas": collect("pred_mano_params", "betas"),
        "focal_length": _linear_fill(
            np.asarray(
                [float(by_frame[int(fid)]["focal_length"]) if int(fid) in by_frame else np.nan for fid in full_frame_ids],
                dtype=np.float32,
            )[:, None],
            observed_mask,
        )[:, 0],
        "img_res": _linear_fill(
            np.stack(
                [
                    np.asarray(by_frame[int(fid)]["img_res"], dtype=np.float32) if int(fid) in by_frame
                    else np.asarray(template["img_res"], dtype=np.float32) * np.nan
                    for fid in full_frame_ids
                ]
            ).astype(np.float32),
            observed_mask,
        ).astype(np.int32),
    }

    if image_folder is not None and video_name is not None:
        image_dir = Path(image_folder)
        if image_dir.name != f"{video_name}_frames":
            image_dir = image_dir / f"{video_name}_frames"
        seq["image_paths"] = [
            image_dir / f"frame_{int(frame_id):06d}.jpg"
            for frame_id in full_frame_ids
        ]

    return seq


def _masked_mean(loss_per_frame: torch.Tensor, mask: torch.Tensor):
    weights = mask.float()
    return (loss_per_frame * weights).sum() / weights.sum().clamp_min(1.0)


def _fft_band_loss(pred_joints, obs_joints, mask, fps, low_hz, high_hz):
    if low_hz is None or high_hz is None or low_hz >= high_hz or pred_joints.shape[0] < 4:
        return pred_joints.new_zeros(())
    signal_pred = pred_joints[:, DEFAULT_FFT_JOINTS] - pred_joints[:, DEFAULT_FFT_JOINTS].mean(dim=0, keepdim=True)
    signal_obs = obs_joints[:, DEFAULT_FFT_JOINTS] - obs_joints[:, DEFAULT_FFT_JOINTS].mean(dim=0, keepdim=True)
    pred_fft = torch.fft.rfft(signal_pred, dim=0)
    obs_fft = torch.fft.rfft(signal_obs, dim=0)
    freqs = torch.fft.rfftfreq(pred_joints.shape[0], d=1.0 / float(fps)).to(pred_joints.device)
    band = (freqs >= low_hz) & (freqs <= high_hz)
    if not bool(band.any().item()):
        return pred_joints.new_zeros(())
    pred_mag = pred_fft.abs()[band]
    obs_mag = obs_fft.abs()[band]
    weight = mask.float().mean().clamp_min(1e-6)
    return F.mse_loss(pred_mag, obs_mag) * weight


def _optimize_sequence(sequence, config: StrideConfig, mano_model_path: str, device: torch.device):
    dtype = torch.float32
    observed_mask = torch.as_tensor(sequence["observed_mask"], device=device, dtype=dtype)
    conf = torch.as_tensor(sequence["detection_confidence"], device=device, dtype=dtype).clamp_min(0.05)

    init_global = torch.as_tensor(sequence["global_orient"], device=device, dtype=dtype)
    init_pose = torch.as_tensor(sequence["hand_pose"], device=device, dtype=dtype)
    init_cam_t = torch.as_tensor(sequence["cam_t"], device=device, dtype=dtype)
    init_box_center = torch.as_tensor(sequence["box_center"], device=device, dtype=dtype)
    init_box_size = torch.as_tensor(sequence["box_size"], device=device, dtype=dtype)
    init_betas = torch.as_tensor(sequence["betas"], device=device, dtype=dtype)
    obs_joints = torch.as_tensor(sequence["joints"], device=device, dtype=dtype)
    obs_kp2d = torch.as_tensor(sequence["pred_keypoints_2d"], device=device, dtype=dtype)
    focal = torch.as_tensor(sequence["focal_length"], device=device, dtype=dtype)
    img_res = torch.as_tensor(sequence["img_res"], device=device, dtype=dtype)

    init_rot6d = torch.cat([_rotmat_to_rot6d(init_global), _rotmat_to_rot6d(init_pose)], dim=1)
    time_len, joint_count = init_rot6d.shape[:2]
    pose_dim = joint_count * 6

    pose_basis = _build_temporal_basis(time_len, config.pose_rank, device, dtype)
    cam_basis = _build_temporal_basis(time_len, config.cam_rank, device, dtype)

    pose_coeff = torch.nn.Parameter(torch.zeros((pose_basis.shape[1], pose_dim), device=device, dtype=dtype))
    cam_coeff = torch.nn.Parameter(torch.zeros((cam_basis.shape[1], 3), device=device, dtype=dtype))
    box_center_coeff = torch.nn.Parameter(torch.zeros((cam_basis.shape[1], 2), device=device, dtype=dtype))
    box_size_coeff = torch.nn.Parameter(torch.zeros((cam_basis.shape[1], 1), device=device, dtype=dtype))
    betas_shared = torch.nn.Parameter(init_betas.mean(dim=0))

    optimizer = torch.optim.Adam([pose_coeff, cam_coeff, box_center_coeff, box_size_coeff, betas_shared], lr=config.lr)

    mano = MANO(
        model_path=mano_model_path,
        batch_size=time_len,
        create_body_pose=False,
    ).to(device)

    best_state = None
    best_loss = None
    last_stats = {}

    for _ in range(config.iters):
        optimizer.zero_grad(set_to_none=True)

        pose_residual = (pose_basis @ pose_coeff).reshape(time_len, joint_count, 6)
        cam_residual = cam_basis @ cam_coeff
        center_residual = cam_basis @ box_center_coeff
        size_residual = cam_basis @ box_size_coeff

        pose6d = init_rot6d + pose_residual
        rotmats = rot6d_to_rotmat(pose6d.reshape(-1, 6)).reshape(time_len, joint_count, 3, 3)
        global_orient = rotmats[:, :1]
        hand_pose = rotmats[:, 1:]
        betas = betas_shared.unsqueeze(0).expand(time_len, -1)
        cam_t = init_cam_t + cam_residual
        box_center = init_box_center + center_residual
        box_size = (init_box_size.unsqueeze(-1) + size_residual).squeeze(-1).clamp_min(1.0)

        mano_out = mano(
            global_orient=global_orient,
            hand_pose=hand_pose,
            betas=betas,
            pose2rot=False,
        )
        verts = mano_out.vertices
        joints = mano_out.joints
        mirror = 2 * int(sequence["right"]) - 1
        joints = joints.clone()
        verts = verts.clone()
        joints[..., 0] *= mirror
        verts[..., 0] *= mirror

        camera_center = torch.stack([img_res[:, 0] / 2.0, img_res[:, 1] / 2.0], dim=-1)
        focal_xy = torch.stack([focal, focal], dim=-1)
        pred_kp2d = perspective_projection(
            joints,
            translation=cam_t,
            focal_length=focal_xy,
            camera_center=camera_center,
        )

        weight = observed_mask * conf
        joints_loss = ((joints - obs_joints).square().mean(dim=(1, 2))) * weight
        reproj_loss = ((pred_kp2d - obs_kp2d).square().mean(dim=(1, 2))) * weight

        cam_second = _second_difference(cam_t)
        pose_second = _second_difference(pose6d)
        joint_second = _second_difference(joints)
        center_second = _second_difference(box_center)
        size_second = _second_difference(box_size)

        loss_obs = joints_loss.sum() / weight.sum().clamp_min(1.0)
        loss_reproj = reproj_loss.sum() / weight.sum().clamp_min(1.0)
        loss_shape = F.mse_loss(betas, betas_shared.unsqueeze(0).expand_as(betas))
        loss_cam_smooth = (
            cam_second.square().mean() if cam_second.numel() else cam_t.new_zeros(())
        ) + 0.1 * (
            center_second.square().mean() if center_second.numel() else cam_t.new_zeros(())
        ) + 0.1 * (
            size_second.square().mean() if size_second.numel() else cam_t.new_zeros(())
        )
        loss_pose_smooth = pose_second.square().mean() if pose_second.numel() else cam_t.new_zeros(())
        loss_joint_smooth = joint_second.square().mean() if joint_second.numel() else cam_t.new_zeros(())
        loss_anchor = ((pose6d - init_rot6d).square().mean(dim=(1, 2)) * weight).sum() / weight.sum().clamp_min(1.0)
        loss_fft = _fft_band_loss(
            joints,
            obs_joints,
            observed_mask,
            config.fps,
            config.fft_band_low_hz,
            config.fft_band_high_hz,
        )

        total = (
            config.obs_weight * loss_obs
            + config.reproj_weight * loss_reproj
            + config.shape_weight * loss_shape
            + config.cam_smooth_weight * loss_cam_smooth
            + config.pose_smooth_weight * loss_pose_smooth
            + config.joint_smooth_weight * loss_joint_smooth
            + config.anchor_weight * loss_anchor
            + config.fft_weight * loss_fft
        )
        total.backward()
        optimizer.step()

        value = float(total.detach().cpu().item())
        if best_loss is None or value < best_loss:
            best_loss = value
            best_state = {
                "global_orient": global_orient.detach().cpu(),
                "hand_pose": hand_pose.detach().cpu(),
                "betas": betas.detach().cpu(),
                "cam_t": cam_t.detach().cpu(),
                "box_center": box_center.detach().cpu(),
                "box_size": box_size.detach().cpu(),
                "joints": joints.detach().cpu(),
                "verts": verts.detach().cpu(),
                "pred_keypoints_2d": pred_kp2d.detach().cpu(),
            }
            last_stats = {
                "total": value,
                "obs": float(loss_obs.detach().cpu().item()),
                "reproj": float(loss_reproj.detach().cpu().item()),
                "shape": float(loss_shape.detach().cpu().item()),
                "cam_smooth": float(loss_cam_smooth.detach().cpu().item()),
                "pose_smooth": float(loss_pose_smooth.detach().cpu().item()),
                "joint_smooth": float(loss_joint_smooth.detach().cpu().item()),
                "anchor": float(loss_anchor.detach().cpu().item()),
                "fft": float(loss_fft.detach().cpu().item()),
            }

    if best_state is None:
        raise RuntimeError("STRIDE optimization did not produce any state.")

    return {
        "frame_ids": sequence["frame_ids"],
        "observed_mask": sequence["observed_mask"],
        "right": sequence["right"],
        "img_res": sequence["img_res"],
        "focal_length": sequence["focal_length"],
        "stats": last_stats,
        **{key: value.numpy() for key, value in best_state.items()},
    }


def _save_outputs(video_name, sequence, refined, output_root: Path, visualize: bool, mano_faces):
    base_dir = output_root / video_name
    mesh_root = base_dir / "meshes"
    vis_root = base_dir / "visualizations"
    mesh_root.mkdir(parents=True, exist_ok=True)
    if visualize:
        vis_root.mkdir(parents=True, exist_ok=True)

    frame_rows = []
    for index, frame_id in enumerate(refined["frame_ids"]):
        frame_name = f"frame_{int(frame_id):06d}"
        frame_dir = mesh_root / frame_name
        frame_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "verts": refined["verts"][index].astype(np.float32),
            "joints": refined["joints"][index].astype(np.float32),
            "cam_t": refined["cam_t"][index].astype(np.float32),
            "right": np.asarray(refined["right"], dtype=np.float32),
            "box_center": refined["box_center"][index].astype(np.float32),
            "box_size": np.asarray(refined["box_size"][index], dtype=np.float32),
            "focal_length": float(refined["focal_length"][index]),
            "img_res": np.asarray(refined["img_res"][index], dtype=np.int32),
            "frame_id": np.asarray(int(frame_id), dtype=np.int32),
            "track_id": np.asarray(0, dtype=np.int32),
            "pred_keypoints_2d": refined["pred_keypoints_2d"][index].astype(np.float32),
            "pred_mano_params": {
                "global_orient": refined["global_orient"][index].astype(np.float32),
                "hand_pose": refined["hand_pose"][index].astype(np.float32),
                "betas": refined["betas"][index].astype(np.float32),
            },
            "observed": np.asarray(bool(refined["observed_mask"][index])),
            "refined": np.asarray(True),
        }
        np.save(frame_dir / f"{frame_name}_0_{float(refined['right']):.1f}_verts.npy", payload)
        frame_rows.append(int(frame_id))

        if visualize and sequence.get("image_paths"):
            import cv2

            image_path = sequence["image_paths"][index]
            if image_path.exists():
                img_cv2 = cv2.imread(str(image_path))
                if img_cv2 is not None:
                    cam_view = render_rgba_multiple(
                        [payload["verts"]],
                        mano_faces,
                        cam_t=[payload["cam_t"]],
                        render_res=tuple(payload["img_res"]),
                        is_right=[refined["right"]],
                        mesh_base_color=LIGHT_PURPLE,
                        scene_bg_color=(1, 1, 1),
                        focal_length=payload["focal_length"],
                    )
                    input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
                    input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
                    overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
                    cv2.imwrite(str(vis_root / f"{frame_name}.jpg"), (255 * overlay[:, :, ::-1]).astype(np.uint8))

    np.savez(
        base_dir / "refined_sequence.npz",
        frame_id=np.asarray(frame_rows, dtype=np.int32),
        observed_mask=np.asarray(refined["observed_mask"], dtype=bool),
        cam_t=refined["cam_t"].astype(np.float32),
        box_center=refined["box_center"].astype(np.float32),
        box_size=refined["box_size"].astype(np.float32),
        global_orient=refined["global_orient"].astype(np.float32),
        hand_pose=refined["hand_pose"].astype(np.float32),
        betas=refined["betas"].astype(np.float32),
        joints=refined["joints"].astype(np.float32),
        verts=refined["verts"].astype(np.float32),
        pred_keypoints_2d=refined["pred_keypoints_2d"].astype(np.float32),
    )
    np.savez(
        base_dir / "camera_poses.npz",
        frame_id=np.asarray(frame_rows, dtype=np.int32),
        cam_t=refined["cam_t"].astype(np.float32),
        focal_length=np.asarray(refined["focal_length"], dtype=np.float32),
        img_res=np.asarray(refined["img_res"], dtype=np.int32),
    )
    cameras_json = {
        "rotation": np.tile(np.eye(3, dtype=np.float32).reshape(1, 9), (len(frame_rows), 1)).tolist(),
        "translation": refined["cam_t"].astype(np.float32).tolist(),
        "intrinsics": np.stack(
            [
                refined["focal_length"].astype(np.float32),
                refined["focal_length"].astype(np.float32),
                refined["img_res"][:, 0].astype(np.float32) / 2.0,
                refined["img_res"][:, 1].astype(np.float32) / 2.0,
            ],
            axis=-1,
        ).tolist(),
    }
    with open(base_dir / "cameras.json", "w", encoding="utf-8") as handle:
        json.dump(cameras_json, handle, indent=2)

    track_info = {
        "tracks": {
            "0": {
                "index": 0,
                "vis_mask": [bool(value) for value in refined["observed_mask"]],
            }
        },
        "meta": {
            "seq_interval": [int(refined["frame_ids"][0]), int(refined["frame_ids"][-1]) + 1],
            "data_interval": [int(refined["frame_ids"][0]), int(refined["frame_ids"][-1]) + 1],
        },
    }
    with open(base_dir / "track_info.json", "w", encoding="utf-8") as handle:
        json.dump(track_info, handle, indent=2)

    metadata = {
        "video": video_name,
        "frames_refined": int(len(refined["frame_ids"])),
        "right": int(refined["right"]),
        "optimization": refined["stats"],
    }
    with open(base_dir / "stride_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    if visualize and vis_root.exists():
        video_root = output_root / "videos"
        video_root.mkdir(parents=True, exist_ok=True)
        images_to_video(vis_root, video_root / f"{video_name}.mp4", fps=30)


def run_stride_refinement(
    source_root,
    output_root,
    video_name,
    image_folder=None,
    visualize=False,
    target_hand="auto",
    mano_model_path="./mano_data",
    use_gpu=True,
    stride_config: StrideConfig | None = None,
):
    source_root = Path(source_root)
    output_root = Path(output_root)
    mesh_root = source_root / video_name / "meshes"
    if not mesh_root.is_dir():
        raise FileNotFoundError(f"WiLoR mesh cache not found: {mesh_root}")

    records_by_frame = _frame_records(mesh_root)
    selected_records = _pick_track(records_by_frame, target_hand=_parse_target_hand(target_hand))
    sequence = _stack_records(selected_records, image_folder=image_folder, video_name=video_name)

    config = stride_config or StrideConfig()
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    refined = _optimize_sequence(sequence, config, mano_model_path=mano_model_path, device=device)

    mano_faces = MANO(
        model_path=mano_model_path,
        batch_size=1,
        create_body_pose=False,
    ).faces
    _save_outputs(video_name, sequence, refined, output_root, visualize, mano_faces)
    return {
        "video": video_name,
        "frames_refined": int(len(refined["frame_ids"])),
        "config": asdict(config),
        "optimization": refined["stats"],
    }
