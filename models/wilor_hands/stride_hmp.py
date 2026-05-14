import json
import sys
from pathlib import Path

ADAPTER_PATH = Path(__file__).resolve().parent
HMP_CLEAN_PATH = ADAPTER_PATH / "hmp_clean"
for import_path in (ADAPTER_PATH, HMP_CLEAN_PATH):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import numpy as np
import torch
import torch.nn.functional as F

from frame_store import FrameStore
from hmp_model_args import Arguments
from nemf.fk import ForwardKinematicsLayer
from nemf.generative import Architecture
from nemf.losses import GeodesicLoss
from rotations import matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix
from stride_config import StrideHMPConfig
from stride_refine import LIGHT_PURPLE, _frame_records, _parse_target_hand, _pick_track, _stack_records
from utils import estimate_angular_velocity, estimate_linear_velocity
from utils_new import render_rgba_multiple
from visualize import images_to_video
from wilor.models import MANO
from wilor.utils.geometry import perspective_projection


def _require_hmp_assets(config: StrideHMPConfig):
    model_config_path = config.hmp_model_config_path
    if not model_config_path.exists():
        raise FileNotFoundError(f"HMP model config not found: {model_config_path}")

    args = Arguments(
        str(ADAPTER_PATH),
        str(model_config_path.parent),
        filename=model_config_path.name,
        mano_dir=config.runtime.mano_model_path,
    )
    required = [
        config.hmp_assets_root / f"mean-{args.data.gender}-{args.data.clip_length}-{args.data.fps}fps.pt",
        config.hmp_assets_root / f"std-{args.data.gender}-{args.data.clip_length}-{args.data.fps}fps.pt",
        config.hmp_assets_root / "results" / "model" / "local_encoder.pth",
        config.hmp_assets_root / "results" / "model" / "nemf.pth",
        config.hmp_assets_root / "results" / "model" / "global_encoder.pth",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required HMP assets for STRIDE refinement:\n" + "\n".join(missing)
        )
    return args


def _wilor_camera_inputs(sequence):
    cam_t = np.asarray(sequence["cam_t"], dtype=np.float32)
    focal = np.asarray(sequence["focal_length"], dtype=np.float32)
    img_res = np.asarray(sequence["img_res"], dtype=np.float32)
    cam_r = np.tile(np.eye(3, dtype=np.float32), (len(sequence["frame_ids"]), 1, 1))
    camera_intrinsics = np.stack(
        [
            focal,
            focal,
            img_res[:, 0] / 2.0,
            img_res[:, 1] / 2.0,
        ],
        axis=-1,
    ).astype(np.float32)
    return {
        "cam_R": cam_r,
        "cam_t": cam_t,
        "intrinsics": camera_intrinsics,
        "img_res": img_res.astype(np.int32),
        "box_center": np.asarray(sequence["box_center"], dtype=np.float32),
        "box_size": np.asarray(sequence["box_size"], dtype=np.float32),
        "focal_length": focal.astype(np.float32),
    }


def _move_hmp_model_to_device(model: Architecture, device: torch.device):
    for module in model.models:
        module.to(device)
    model.device = device
    model.fk.device = device
    model.fk.parents = model.fk.parents.to(device)
    model.fk.positions = model.fk.positions.to(device)
    model.mean = {key: value.to(device) for key, value in model.mean.items()}
    model.std = {key: value.to(device) for key, value in model.std.items()}
    model.eval()


def _load_hmp_model(config: StrideHMPConfig):
    args = Arguments(
        str(ADAPTER_PATH),
        str(config.hmp_model_config_path.parent),
        filename=config.hmp_model_config_path.name,
        mano_dir=config.runtime.mano_model_path,
    )
    args.root = str(ADAPTER_PATH)
    args.dataset_dir = str(config.hmp_assets_root)
    args.save_dir = str(config.hmp_assets_root)
    args.smpl.smpl_body_model = str(config.runtime.mano_model_path)

    model = Architecture(args, ngpu=1)
    model.load(optimal=True)

    device = torch.device("cuda" if config.runtime.use_gpu and torch.cuda.is_available() else "cpu")
    _move_hmp_model_to_device(model, device)
    fk = ForwardKinematicsLayer(args, device=device)
    return args, model, fk, device


def _sequence_to_hmp_targets(sequence, fk: ForwardKinematicsLayer, fps: float, device: torch.device):
    local_root = torch.as_tensor(sequence["global_orient"], dtype=torch.float32, device=device)
    local_hand = torch.as_tensor(sequence["hand_pose"], dtype=torch.float32, device=device)
    local_rotmat = torch.cat([local_root, local_hand], dim=1)
    root_orient = local_root[:, 0].clone()

    rootless_rotmat = local_rotmat.clone()
    identity = torch.eye(3, dtype=torch.float32, device=device)
    rootless_rotmat[:, 0] = identity

    pos, global_xform = fk(rootless_rotmat)
    global_rotmat = global_xform[:, :, :3, :3].contiguous()
    dt = 1.0 / float(fps)

    return {
        "pos": pos.contiguous(),
        "velocity": estimate_linear_velocity(pos.unsqueeze(0), dt=dt).squeeze(0).contiguous(),
        "global_xform": matrix_to_rotation_6d(global_rotmat).contiguous(),
        "angular": estimate_angular_velocity(rootless_rotmat.unsqueeze(0).clone(), dt=dt).squeeze(0).contiguous(),
        "root_orient": matrix_to_rotation_6d(root_orient).contiguous(),
        "rotmat": global_rotmat.contiguous(),
        "observed_mask": torch.as_tensor(sequence["observed_mask"], dtype=torch.bool, device=device),
        "confidence": torch.as_tensor(sequence["detection_confidence"], dtype=torch.float32, device=device).clamp_min(0.05),
    }


def _chunk_starts(length: int, clip_length: int, overlap: int):
    if length <= clip_length:
        return [0]
    stride = max(1, clip_length - overlap)
    starts = list(range(0, max(1, length - clip_length + 1), stride))
    last_start = length - clip_length
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _slice_time_chunk(tensor: torch.Tensor, start: int, chunk_len: int, pad_value=None):
    end = min(tensor.shape[0], start + chunk_len)
    chunk = tensor[start:end]
    pad = chunk_len - chunk.shape[0]
    if pad <= 0:
        return chunk
    if pad_value is not None:
        extra = torch.full(
            (pad, *tensor.shape[1:]),
            pad_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    elif chunk.shape[0] > 0:
        extra = chunk[-1:].expand(pad, *chunk.shape[1:]).clone()
    else:
        extra = tensor.new_zeros((pad, *tensor.shape[1:]))
    return torch.cat([chunk, extra], dim=0)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor):
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(0)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _masked_rotation_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, criterion_geo: GeodesicLoss):
    per_joint = criterion_geo(
        pred.reshape(-1, 3, 3),
        target.reshape(-1, 3, 3),
        reduction="none",
    ).reshape(pred.shape[0], pred.shape[1], pred.shape[2])
    per_frame = per_joint.mean(dim=2)
    return _weighted_mean(per_frame, weights)


def _masked_position_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
    per_frame = (pred - target).square().mean(dim=(2, 3))
    return _weighted_mean(per_frame, weights)


def _masked_root_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, criterion_geo: GeodesicLoss):
    per_frame = criterion_geo(
        pred.reshape(-1, 3, 3),
        target.reshape(-1, 3, 3),
        reduction="none",
    ).reshape(pred.shape[0], pred.shape[1])
    return _weighted_mean(per_frame, weights)


def _build_chunk_targets(base_targets: dict, start: int, clip_length: int):
    return {
        "pos": _slice_time_chunk(base_targets["pos"], start, clip_length).unsqueeze(0),
        "velocity": _slice_time_chunk(base_targets["velocity"], start, clip_length).unsqueeze(0),
        "global_xform": _slice_time_chunk(base_targets["global_xform"], start, clip_length).unsqueeze(0),
        "angular": _slice_time_chunk(base_targets["angular"], start, clip_length).unsqueeze(0),
        "root_orient": _slice_time_chunk(base_targets["root_orient"], start, clip_length).unsqueeze(0),
        "rotmat": _slice_time_chunk(base_targets["rotmat"], start, clip_length).unsqueeze(0),
        "observed_mask": _slice_time_chunk(base_targets["observed_mask"], start, clip_length, pad_value=False),
        "confidence": _slice_time_chunk(base_targets["confidence"], start, clip_length, pad_value=0.0),
    }


def _optimize_pose_chunk(model: Architecture, config: StrideHMPConfig, chunk_targets: dict):
    model_input = {
        "pos": chunk_targets["pos"],
        "velocity": chunk_targets["velocity"],
        "global_xform": chunk_targets["global_xform"],
        "angular": chunk_targets["angular"],
        "root_orient": chunk_targets["root_orient"],
    }
    model.set_input(model_input)
    with torch.no_grad():
        _, init_z_l, _ = model.encode_local()
        _, init_z_g, _ = model.encode_global()

    z_l = torch.nn.Parameter(init_z_l.detach().clone())
    z_g = torch.nn.Parameter(init_z_g.detach().clone())
    optimizer = torch.optim.Adam([z_l, z_g], lr=config.pose_lr)
    criterion_geo = GeodesicLoss()

    weights = chunk_targets["observed_mask"].float() * chunk_targets["confidence"]
    best_total = None
    best_local_rotmat = None
    best_stats = {}

    for _ in range(config.pose_iters):
        optimizer.zero_grad(set_to_none=True)
        output = model.decode(z_l, z_g, length=chunk_targets["rotmat"].shape[1], step=1)

        loss_rot = _masked_rotation_loss(output["rotmat"], chunk_targets["rotmat"], weights, criterion_geo)
        loss_pos = _masked_position_loss(output["pos"], chunk_targets["pos"], weights)
        loss_root = _masked_root_loss(
            rotation_6d_to_matrix(output["root_orient"]),
            rotation_6d_to_matrix(chunk_targets["root_orient"]),
            weights,
            criterion_geo,
        )
        loss_latent = F.mse_loss(z_l, init_z_l) + F.mse_loss(z_g, init_z_g)

        total = (
            config.pose_weights.rot * loss_rot
            + config.pose_weights.pos * loss_pos
            + config.pose_weights.root * loss_root
            + config.pose_weights.latent * loss_latent
        )
        total.backward()
        optimizer.step()

        value = float(total.detach().cpu().item())
        if best_total is None or value < best_total:
            with torch.no_grad():
                best_total = value
                root_orient = rotation_6d_to_matrix(output["root_orient"])
                joint_count = output["rotmat"].shape[2]
                local_rotmat = model.fk.global_to_local(output["rotmat"].reshape(-1, joint_count, 3, 3)).reshape(
                    output["rotmat"].shape[0],
                    output["rotmat"].shape[1],
                    joint_count,
                    3,
                    3,
                )
                local_rotmat[:, :, 0] = root_orient
                best_local_rotmat = local_rotmat[0].detach().cpu()
                best_stats = {
                    "total": value,
                    "rot": float(loss_rot.detach().cpu().item()),
                    "pos": float(loss_pos.detach().cpu().item()),
                    "root": float(loss_root.detach().cpu().item()),
                    "latent": float(loss_latent.detach().cpu().item()),
                }

    if best_local_rotmat is None:
        raise RuntimeError("HMP pose optimization did not produce a result.")
    return best_local_rotmat, best_stats


def _blend_pose_chunks(chunk_rotmats: list[torch.Tensor], starts: list[int], seq_len: int, overlap: int, device: torch.device):
    joint_count = chunk_rotmats[0].shape[1]
    accum = torch.zeros((seq_len, joint_count, 6), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((seq_len, 1, 1), dtype=torch.float32, device=device)

    for chunk_rotmat, start in zip(chunk_rotmats, starts):
        valid_len = min(seq_len - start, chunk_rotmat.shape[0])
        weights = torch.ones((valid_len,), dtype=torch.float32, device=device)
        if overlap > 0 and valid_len > 1:
            ramp_len = min(overlap, valid_len)
            ramp = torch.linspace(0.25, 1.0, steps=ramp_len, device=device)
            weights[:ramp_len] = ramp
            weights[-ramp_len:] = torch.minimum(weights[-ramp_len:], ramp.flip(0))

        rot6d = matrix_to_rotation_6d(chunk_rotmat[:valid_len].to(device))
        span = slice(start, start + valid_len)
        accum[span] += rot6d * weights[:, None, None]
        weight_sum[span] += weights[:, None, None]

    blended6d = accum / weight_sum.clamp_min(1e-6)
    return rotation_6d_to_matrix(blended6d.reshape(-1, 6)).reshape(seq_len, joint_count, 3, 3)


def _optimize_sequence_betas(sequence, local_rotmat: torch.Tensor, config: StrideHMPConfig, mano_model_path: Path, device: torch.device):
    seq_len = local_rotmat.shape[0]
    init_betas = torch.as_tensor(sequence["betas"], dtype=torch.float32, device=device)
    mean_betas = init_betas.mean(dim=0)
    if not config.beta_optimize or config.beta_iters <= 0:
        return mean_betas.unsqueeze(0).expand(seq_len, -1).detach().cpu(), {
            "enabled": False,
            "joint": 0.0,
            "prior": 0.0,
            "total": 0.0,
        }

    obs_joints = torch.as_tensor(sequence["joints"], dtype=torch.float32, device=device)
    weights = (
        torch.as_tensor(sequence["observed_mask"], dtype=torch.float32, device=device)
        * torch.as_tensor(sequence["detection_confidence"], dtype=torch.float32, device=device).clamp_min(0.05)
    )
    beta_param = torch.nn.Parameter(mean_betas.clone())
    optimizer = torch.optim.Adam([beta_param], lr=config.beta_lr)
    mano = MANO(
        model_path=str(mano_model_path),
        batch_size=seq_len,
        create_body_pose=False,
        use_pca=False,
    ).to(device)
    mirror = 2 * int(sequence["right"]) - 1

    best_total = None
    best_betas = None
    best_stats = {}
    global_orient = local_rotmat[:, :1].to(device)
    hand_pose = local_rotmat[:, 1:].to(device)

    for _ in range(config.beta_iters):
        optimizer.zero_grad(set_to_none=True)
        betas = beta_param.unsqueeze(0).expand(seq_len, -1)
        mano_out = mano(global_orient=global_orient, hand_pose=hand_pose, betas=betas, pose2rot=False)
        joints = mano_out.joints.clone()
        joints[..., 0] *= mirror

        per_frame = (joints - obs_joints).square().mean(dim=(1, 2))
        loss_joint = _weighted_mean(per_frame.unsqueeze(0), weights)
        loss_prior = F.mse_loss(beta_param, mean_betas)
        total = config.beta_weights.joint * loss_joint + config.beta_weights.prior * loss_prior
        total.backward()
        optimizer.step()

        value = float(total.detach().cpu().item())
        if best_total is None or value < best_total:
            best_total = value
            best_betas = betas.detach().cpu()
            best_stats = {
                "enabled": True,
                "joint": float(loss_joint.detach().cpu().item()),
                "prior": float(loss_prior.detach().cpu().item()),
                "total": value,
            }

    if best_betas is None:
        raise RuntimeError("Beta optimization did not produce a result.")
    return best_betas, best_stats


def _reconstruct_sequence(sequence, local_rotmat: torch.Tensor, betas: torch.Tensor, camera_inputs: dict, mano_model_path: Path, device: torch.device):
    seq_len = local_rotmat.shape[0]
    global_orient = local_rotmat[:, :1].to(device)
    hand_pose = local_rotmat[:, 1:].to(device)
    betas_t = betas.to(device)

    mano = MANO(
        model_path=str(mano_model_path),
        batch_size=seq_len,
        create_body_pose=False,
        use_pca=False,
    ).to(device)

    cam_t = torch.as_tensor(camera_inputs["cam_t"], dtype=torch.float32, device=device)
    camera_intrinsics = torch.as_tensor(camera_inputs["intrinsics"], dtype=torch.float32, device=device)
    focal_xy = camera_intrinsics[:, :2]
    camera_center = camera_intrinsics[:, 2:]
    mirror = 2 * int(sequence["right"]) - 1

    with torch.no_grad():
        mano_out = mano(global_orient=global_orient, hand_pose=hand_pose, betas=betas_t, pose2rot=False)
        verts = mano_out.vertices.clone()
        joints = mano_out.joints.clone()
        verts[..., 0] *= mirror
        joints[..., 0] *= mirror
        kp2d = perspective_projection(
            joints,
            translation=cam_t,
            focal_length=focal_xy,
            camera_center=camera_center,
        )

    root_orient_aa = matrix_to_axis_angle(global_orient[:, 0]).detach().cpu().numpy().astype(np.float32)
    pose_body_aa = matrix_to_axis_angle(hand_pose.reshape(-1, 3, 3)).reshape(seq_len, 15, 3).detach().cpu().numpy().astype(np.float32)

    return {
        "frame_ids": np.asarray(sequence["frame_ids"], dtype=np.int32),
        "observed_mask": np.asarray(sequence["observed_mask"], dtype=bool),
        "right": int(sequence["right"]),
        "img_res": np.asarray(camera_inputs["img_res"], dtype=np.int32),
        "box_center": np.asarray(camera_inputs["box_center"], dtype=np.float32),
        "box_size": np.asarray(camera_inputs["box_size"], dtype=np.float32),
        "focal_length": np.asarray(camera_inputs["focal_length"], dtype=np.float32),
        "camera_intrinsics": np.asarray(camera_inputs["intrinsics"], dtype=np.float32),
        "cam_t": np.asarray(camera_inputs["cam_t"], dtype=np.float32),
        "cam_R": np.asarray(camera_inputs["cam_R"], dtype=np.float32),
        "global_orient": global_orient.detach().cpu().numpy().astype(np.float32),
        "hand_pose": hand_pose.detach().cpu().numpy().astype(np.float32),
        "betas": betas_t.detach().cpu().numpy().astype(np.float32),
        "joints": joints.detach().cpu().numpy().astype(np.float32),
        "verts": verts.detach().cpu().numpy().astype(np.float32),
        "pred_keypoints_2d": kp2d.detach().cpu().numpy().astype(np.float32),
        "root_orient_aa": root_orient_aa,
        "pose_body_aa": pose_body_aa,
    }


def _save_raw_pose_result(path: Path, refined: dict):
    seq_len = len(refined["frame_ids"])
    np.savez(
        path,
        root_orient=refined["root_orient_aa"][None],
        pose_body=refined["pose_body_aa"][None],
        betas=refined["betas"][None],
        cam_t=refined["cam_t"][None],
        cam_R=refined["cam_R"][None],
        intrins=refined["camera_intrinsics"][None],
        is_right=np.full((1, seq_len), refined["right"], dtype=np.float32),
    )


def _save_outputs(video_name, sequence, refined, output_root: Path, raw_result_path: Path, visualize: bool, mano_faces, frame_store: FrameStore | None = None):
    base_dir = output_root / video_name
    mesh_root = base_dir / "meshes"
    vis_root = base_dir / "visualizations"
    mesh_root.mkdir(parents=True, exist_ok=True)
    if visualize:
        vis_root.mkdir(parents=True, exist_ok=True)

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
            "backend": np.asarray("hmp"),
        }
        np.save(frame_dir / f"{frame_name}_0_{float(refined['right']):.1f}_verts.npy", payload)

        if visualize and frame_store is not None and sequence.get("video_name"):
            import cv2

            img_cv2 = frame_store.get_frame(str(sequence["video_name"]), int(frame_id))
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

    _save_raw_pose_result(raw_result_path, refined)

    np.savez(
        base_dir / "refined_sequence.npz",
        frame_id=np.asarray(refined["frame_ids"], dtype=np.int32),
        observed_mask=np.asarray(refined["observed_mask"], dtype=bool),
        cam_t=refined["cam_t"].astype(np.float32),
        cam_R=refined["cam_R"].astype(np.float32),
        camera_intrinsics=refined["camera_intrinsics"].astype(np.float32),
        box_center=refined["box_center"].astype(np.float32),
        box_size=refined["box_size"].astype(np.float32),
        global_orient=refined["global_orient"].astype(np.float32),
        hand_pose=refined["hand_pose"].astype(np.float32),
        betas=refined["betas"].astype(np.float32),
        joints=refined["joints"].astype(np.float32),
        verts=refined["verts"].astype(np.float32),
        pred_keypoints_2d=refined["pred_keypoints_2d"].astype(np.float32),
        root_orient_aa=refined["root_orient_aa"].astype(np.float32),
        pose_body_aa=refined["pose_body_aa"].astype(np.float32),
    )
    np.savez(
        base_dir / "camera_poses.npz",
        frame_id=np.asarray(refined["frame_ids"], dtype=np.int32),
        cam_R=refined["cam_R"].astype(np.float32),
        cam_t=refined["cam_t"].astype(np.float32),
        intrinsics=refined["camera_intrinsics"].astype(np.float32),
        focal_length=np.asarray(refined["focal_length"], dtype=np.float32),
        img_res=np.asarray(refined["img_res"], dtype=np.int32),
    )
    cameras_json = {
        "rotation": refined["cam_R"].reshape(len(refined["frame_ids"]), 9).astype(np.float32).tolist(),
        "translation": refined["cam_t"].astype(np.float32).tolist(),
        "intrinsics": refined["camera_intrinsics"].astype(np.float32).tolist(),
    }
    with open(base_dir / "cameras.json", "w", encoding="utf-8") as handle:
        json.dump(cameras_json, handle, indent=2)

    track_info = {
        "tracks": {
            "0": {
                "index": 0,
                "right": int(refined["right"]),
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

    if visualize and vis_root.exists():
        video_root = output_root / "videos"
        video_root.mkdir(parents=True, exist_ok=True)
        images_to_video(vis_root, video_root / f"{video_name}.mp4", fps=30)


def run_stride_hmp(
    cache_root,
    output_root,
    video_name,
    frame_store: FrameStore | None = None,
    target_hand="auto",
    hmp_config: StrideHMPConfig | None = None,
):
    if hmp_config is None:
        raise ValueError("run_stride_hmp requires a loaded StrideHMPConfig.")

    cache_root = Path(cache_root)
    output_root = Path(output_root)
    mesh_root = cache_root / video_name / "meshes"
    if not mesh_root.is_dir():
        raise FileNotFoundError(f"WiLoR mesh cache not found: {mesh_root}")

    config = hmp_config
    _require_hmp_assets(config)
    records_by_frame = _frame_records(mesh_root)
    selected_records = _pick_track(records_by_frame, target_hand=_parse_target_hand(target_hand))
    sequence = _stack_records(selected_records, video_name=video_name)
    camera_inputs = _wilor_camera_inputs(sequence)

    hmp_args, model, fk, device = _load_hmp_model(config)
    targets = _sequence_to_hmp_targets(sequence, fk=fk, fps=hmp_args.data.fps, device=device)
    seq_len = len(sequence["frame_ids"])
    overlap = int(config.overlap if config.overlap is not None else hmp_args.overlap_len)
    clip_length = int(hmp_args.data.clip_length)
    starts = _chunk_starts(seq_len, clip_length, overlap)

    chunk_rotmats = []
    pose_stats = []
    for start in starts:
        chunk_targets = _build_chunk_targets(targets, start=start, clip_length=clip_length)
        chunk_rotmat, stats = _optimize_pose_chunk(model, config, chunk_targets)
        chunk_rotmats.append(chunk_rotmat)
        pose_stats.append(stats)

    blended_local_rotmat = _blend_pose_chunks(chunk_rotmats, starts, seq_len, overlap, device=device)
    betas, beta_stats = _optimize_sequence_betas(
        sequence=sequence,
        local_rotmat=blended_local_rotmat,
        config=config,
        mano_model_path=config.runtime.mano_model_path,
        device=device,
    )
    refined = _reconstruct_sequence(
        sequence=sequence,
        local_rotmat=blended_local_rotmat,
        betas=betas,
        camera_inputs=camera_inputs,
        mano_model_path=config.runtime.mano_model_path,
        device=device,
    )

    out_dir = output_root / video_name
    out_dir.mkdir(parents=True, exist_ok=True)
    mano_faces = MANO(
        model_path=str(config.runtime.mano_model_path),
        batch_size=1,
        create_body_pose=False,
        use_pca=False,
    ).faces
    raw_result_path = out_dir / "refined_world_results.npz"
    _save_outputs(
        video_name,
        sequence,
        refined,
        output_root,
        raw_result_path,
        visualize=config.runtime.visualize,
        mano_faces=mano_faces,
        frame_store=frame_store,
    )

    pose_summary = {
        key: float(np.mean([stats[key] for stats in pose_stats])) for key in pose_stats[0]
    } if pose_stats else {}
    metadata = {
        "video": video_name,
        "frames_refined": int(len(refined["frame_ids"])),
        "backend": "hmp",
        "mode": "stride",
        "camera_source": "wilor",
        "source_cache_root": str(cache_root),
        "source_mesh_root": str(mesh_root),
        "stride_config_path": str(config.config_path),
        "hmp_assets_root": str(config.hmp_assets_root),
        "hmp_model_config": str(config.hmp_model_config_path),
        "target_hand": target_hand,
        "right": int(refined["right"]),
        "result_path": str(raw_result_path),
        "clip_length": clip_length,
        "chunk_count": len(starts),
        "pose_optimization": pose_summary,
        "beta_optimization": beta_stats,
    }
    with open(out_dir / "stride_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata
