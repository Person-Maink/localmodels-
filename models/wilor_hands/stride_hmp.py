import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from hmp_prior.arguments import Arguments
from hmp_prior.fitting import fitting_prior
from hmp_prior.rotations import axis_angle_to_matrix, matrix_to_axis_angle
from stride_refine import LIGHT_PURPLE, _frame_records, _parse_target_hand, _pick_track, _stack_records
from utils_new import render_rgba_multiple
from visualize import images_to_video
from wilor.models import MANO
from wilor.utils.geometry import perspective_projection


@dataclass
class HMPConfig:
    assets_root: str
    config_name: str = "hmp_config.yaml"


def _require_hmp_assets(assets_root: Path, config_name: str):
    config = Arguments(
        str(Path(__file__).resolve().parent),
        str(Path(__file__).resolve().parent / "hmp_prior"),
        filename=config_name,
    )
    required = [
        assets_root / f"mean-{config.data.gender}-{config.data.clip_length}-{config.data.fps}fps.pt",
        assets_root / f"std-{config.data.gender}-{config.data.clip_length}-{config.data.fps}fps.pt",
        assets_root / "results" / "model" / "local_encoder.pth",
        assets_root / "results" / "model" / "nemf.pth",
        assets_root / "results" / "model" / "global_encoder.pth",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required HMP assets for STRIDE refinement:\n" + "\n".join(missing)
        )


def _resolve_vid_path(image_folder, cache_root: Path, video_name: str):
    if image_folder:
        image_dir = Path(image_folder)
        if image_dir.name != f"{video_name}_frames":
            image_dir = image_dir / f"{video_name}_frames"
        if image_dir.is_dir():
            return image_dir
    return cache_root / video_name / "meshes"


def _sequence_to_hmp_inputs(sequence, device: torch.device):
    seq_len = len(sequence["frame_ids"])
    right = int(sequence["right"])

    global_orient_rot = torch.as_tensor(sequence["global_orient"], dtype=torch.float32, device=device)
    hand_pose_rot = torch.as_tensor(sequence["hand_pose"], dtype=torch.float32, device=device)

    root_orient = matrix_to_axis_angle(global_orient_rot[:, 0]).detach().cpu().numpy().astype(np.float32)
    pose_body = matrix_to_axis_angle(hand_pose_rot.reshape(-1, 3, 3)).reshape(seq_len, 15, 3).detach().cpu().numpy().astype(np.float32)
    cam_t = np.asarray(sequence["cam_t"], dtype=np.float32)
    betas = np.asarray(sequence["betas"], dtype=np.float32)
    focal = np.asarray(sequence["focal_length"], dtype=np.float32)
    img_res = np.asarray(sequence["img_res"], dtype=np.float32)
    cam_r = np.tile(np.eye(3, dtype=np.float32), (seq_len, 1, 1))

    conf = np.asarray(sequence["detection_confidence"], dtype=np.float32)
    joints2d_xy = np.asarray(sequence["pred_keypoints_2d"], dtype=np.float32)
    joints2d = np.concatenate([joints2d_xy, np.repeat(conf[:, None, None], joints2d_xy.shape[1], axis=1)], axis=-1)
    vis_mask = np.asarray(sequence["observed_mask"], dtype=np.bool_)
    intrins = np.asarray(
        [float(focal.mean()), float(focal.mean()), float(img_res[0, 0] / 2.0), float(img_res[0, 1] / 2.0)],
        dtype=np.float32,
    )

    obs_data = {
        "joints2d": torch.from_numpy(joints2d[None]).to(device),
        "vis_mask": torch.from_numpy(vis_mask[None]).to(device),
        "is_right": torch.from_numpy(np.full((1, seq_len), right, dtype=np.float32)).to(device),
    }
    res_dict = [{
        "pose_body": torch.from_numpy(pose_body[None]).to(device),
        "betas": torch.from_numpy(betas[None]).to(device),
        "trans": torch.from_numpy(cam_t[None]).to(device),
        "root_orient": torch.from_numpy(root_orient[None]).to(device),
        "is_right": torch.from_numpy(np.full((1, seq_len), right, dtype=np.float32)).to(device),
        "cam_R": torch.from_numpy(cam_r[None]).to(device),
        "cam_t": torch.from_numpy(cam_t[None]).to(device),
        "intrins": torch.from_numpy(intrins).to(device),
    }]
    return obs_data, res_dict


def _result_stem(vid_path: Path):
    return vid_path.name.split(".")[0]


def _load_world_result(path: Path):
    payload = np.load(path, allow_pickle=True)
    return {key: payload[key] for key in payload.files}


def _reconstruct_sequence(sequence, result_dict, mano_model_path: str, device: torch.device):
    frame_ids = np.asarray(sequence["frame_ids"], dtype=np.int32)
    observed_mask = np.asarray(sequence["observed_mask"], dtype=bool)
    right = int(sequence["right"])
    img_res = np.asarray(sequence["img_res"], dtype=np.int32)
    box_center = np.asarray(sequence["box_center"], dtype=np.float32)
    box_size = np.asarray(sequence["box_size"], dtype=np.float32)
    focal = np.asarray(sequence["focal_length"], dtype=np.float32)

    root_orient_aa = np.asarray(result_dict["root_orient"], dtype=np.float32)[0]
    pose_body_aa = np.asarray(result_dict["pose_body"], dtype=np.float32)[0]
    cam_t = np.asarray(result_dict["cam_t"], dtype=np.float32)[0]
    cam_r = np.asarray(result_dict["cam_R"], dtype=np.float32)[0]
    betas = np.asarray(result_dict["betas"], dtype=np.float32)
    if betas.ndim == 2:
        betas = np.repeat(betas, len(frame_ids), axis=0)
    else:
        betas = np.repeat(betas[0][None], len(frame_ids), axis=0)

    seq_len = min(len(frame_ids), len(root_orient_aa), len(pose_body_aa), len(cam_t))
    frame_ids = frame_ids[:seq_len]
    observed_mask = observed_mask[:seq_len]
    img_res = img_res[:seq_len]
    box_center = box_center[:seq_len]
    box_size = box_size[:seq_len]
    focal = focal[:seq_len]
    root_orient_aa = root_orient_aa[:seq_len]
    pose_body_aa = pose_body_aa[:seq_len]
    cam_t = cam_t[:seq_len]
    cam_r = cam_r[:seq_len]
    betas = betas[:seq_len]

    mano = MANO(
        model_path=mano_model_path,
        batch_size=seq_len,
        create_body_pose=False,
    ).to(device)
    with torch.no_grad():
        root_orient_t = torch.from_numpy(root_orient_aa).to(device)
        pose_body_t = torch.from_numpy(pose_body_aa.reshape(seq_len, -1)).to(device)
        betas_t = torch.from_numpy(betas).to(device)
        mano_out = mano(global_orient=root_orient_t, hand_pose=pose_body_t, betas=betas_t)
        verts = mano_out.vertices.detach().cpu().numpy().astype(np.float32)
        joints = mano_out.joints.detach().cpu().numpy().astype(np.float32)

        mirror = 2 * right - 1
        verts[..., 0] *= mirror
        joints[..., 0] *= mirror

        kp2d = perspective_projection(
            torch.from_numpy(joints).to(device),
            translation=torch.from_numpy(cam_t).to(device),
            focal_length=torch.from_numpy(np.stack([focal, focal], axis=-1)).to(device),
            camera_center=torch.from_numpy(np.stack([img_res[:, 0] / 2.0, img_res[:, 1] / 2.0], axis=-1).astype(np.float32)).to(device),
        ).detach().cpu().numpy().astype(np.float32)

    global_rot = axis_angle_to_matrix(torch.from_numpy(root_orient_aa)).numpy().astype(np.float32)[:, None]
    hand_rot = axis_angle_to_matrix(torch.from_numpy(pose_body_aa.reshape(-1, 3))).numpy().astype(np.float32).reshape(seq_len, 15, 3, 3)

    return {
        "frame_ids": frame_ids,
        "observed_mask": observed_mask,
        "right": right,
        "img_res": img_res,
        "box_center": box_center,
        "box_size": box_size,
        "focal_length": focal,
        "cam_t": cam_t.astype(np.float32),
        "cam_R": cam_r.astype(np.float32),
        "global_orient": global_rot,
        "hand_pose": hand_rot,
        "betas": betas.astype(np.float32),
        "joints": joints,
        "verts": verts,
        "pred_keypoints_2d": kp2d,
        "root_orient_aa": root_orient_aa.astype(np.float32),
        "pose_body_aa": pose_body_aa.astype(np.float32),
    }


def _save_outputs(video_name, sequence, refined, output_root: Path, raw_result_path: Path, visualize: bool, mano_faces):
    base_dir = output_root / video_name
    mesh_root = base_dir / "meshes"
    vis_root = base_dir / "visualizations"
    mesh_root.mkdir(parents=True, exist_ok=True)
    if visualize:
        vis_root.mkdir(parents=True, exist_ok=True)

    shutil.copy2(raw_result_path, base_dir / "refined_world_results.npz")

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
        frame_id=np.asarray(refined["frame_ids"], dtype=np.int32),
        observed_mask=np.asarray(refined["observed_mask"], dtype=bool),
        cam_t=refined["cam_t"].astype(np.float32),
        cam_R=refined["cam_R"].astype(np.float32),
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
        focal_length=np.asarray(refined["focal_length"], dtype=np.float32),
        img_res=np.asarray(refined["img_res"], dtype=np.int32),
    )
    cameras_json = {
        "rotation": refined["cam_R"].reshape(len(refined["frame_ids"]), 9).astype(np.float32).tolist(),
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
    image_folder=None,
    target_hand="auto",
    mano_model_path="./mano_data",
    use_gpu=True,
    visualize=False,
    hmp_config: HMPConfig | None = None,
):
    config = hmp_config or HMPConfig(assets_root=str(Path(__file__).resolve().parent / "_DATA" / "hmp_model"))
    assets_root = Path(config.assets_root)
    _require_hmp_assets(assets_root, config.config_name)

    cache_root = Path(cache_root)
    output_root = Path(output_root)
    mesh_root = cache_root / video_name / "meshes"
    if not mesh_root.is_dir():
        raise FileNotFoundError(f"WiLoR mesh cache not found: {mesh_root}")

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    records_by_frame = _frame_records(mesh_root)
    selected_records = _pick_track(records_by_frame, target_hand=_parse_target_hand(target_hand))
    sequence = _stack_records(selected_records, image_folder=image_folder, video_name=video_name)
    obs_data, res_dict = _sequence_to_hmp_inputs(sequence, device)

    out_dir = output_root / video_name
    out_dir.mkdir(parents=True, exist_ok=True)
    vid_path = _resolve_vid_path(image_folder, cache_root, video_name)

    opt = SimpleNamespace(
        paths=SimpleNamespace(base_dir=str(Path(__file__).resolve().parent)),
        HMP=SimpleNamespace(
            config=config.config_name,
            vid_path=str(vid_path),
            exp_name=video_name,
            use_hposer=False,
            assets_root=str(assets_root),
            mano_dir=str(Path(mano_model_path)),
        ),
    )
    data_args = SimpleNamespace(seq=video_name)
    hand_model = MANO(model_path=mano_model_path, batch_size=128, create_body_pose=False).to(device)

    fitting_prior(obs_data, res_dict, hand_model, opt, data_args, str(out_dir), device)

    raw_result_path = out_dir / f"{_result_stem(vid_path)}_000000_world_results.npz"
    if not raw_result_path.exists():
        raise FileNotFoundError(f"HMP refinement did not produce expected output: {raw_result_path}")

    result_dict = _load_world_result(raw_result_path)
    refined = _reconstruct_sequence(sequence, result_dict, mano_model_path=mano_model_path, device=device)
    mano_faces = MANO(model_path=mano_model_path, batch_size=1, create_body_pose=False).faces
    _save_outputs(video_name, sequence, refined, output_root, raw_result_path, visualize, mano_faces)

    metadata = {
        "video": video_name,
        "frames_refined": int(len(refined["frame_ids"])),
        "backend": "hmp",
        "source_cache_root": str(cache_root),
        "source_mesh_root": str(mesh_root),
        "hmp_assets_root": str(assets_root),
        "hmp_config_name": config.config_name,
        "target_hand": target_hand,
        "right": int(refined["right"]),
        "result_path": str(out_dir / "refined_world_results.npz"),
        "raw_result_path": str(raw_result_path),
    }
    with open(out_dir / "stride_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata
