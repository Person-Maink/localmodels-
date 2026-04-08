# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import tempfile
import zipfile

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import imageio
import Imath
import numpy as np
import OpenEXR
import torch

from vipe.ext.lietorch import SE3
from vipe.streams.base import FrameAttribute, VideoFrame, VideoStream
from vipe.utils.cameras import CameraType
from vipe.utils.geometry import se3_matrix_to_se3
from vipe.utils.visualization import VideoWriter


logger = logging.getLogger(__name__)


@dataclass
class ArtifactPath:
    base_path: Path
    artifact_name: str

    @property
    def rgb_path(self) -> Path:
        return self.base_path / "rgb" / f"{self.artifact_name}.mp4"

    @property
    def pose_path(self) -> Path:
        return self.base_path / "pose" / f"{self.artifact_name}.npz"

    @property
    def depth_path(self) -> Path:
        return self.base_path / "depth" / f"{self.artifact_name}.zip"

    @property
    def intrinsics_path(self) -> Path:
        return self.base_path / "intrinsics" / f"{self.artifact_name}.npz"

    @property
    def camera_type_path(self) -> Path:
        return self.base_path / "intrinsics" / f"{self.artifact_name}_camera.txt"

    @property
    def flow_path(self) -> Path:
        return self.base_path / "flow" / f"{self.artifact_name}.zip"

    @property
    def mask_path(self) -> Path:
        return self.base_path / "mask" / f"{self.artifact_name}.zip"

    @property
    def mask_phrase_path(self) -> Path:
        return self.base_path / "mask" / f"{self.artifact_name}.txt"

    @property
    def meta_info_path(self) -> Path:
        return self.base_path / "vipe" / f"{self.artifact_name}_info.pkl"

    @classmethod
    def glob_artifacts(cls, base_path: Path, use_video: bool = False) -> Iterator["ArtifactPath"]:
        if use_video:
            for artifact_path in (base_path / "rgb").glob("*.mp4"):
                artifact_name = artifact_path.stem
                yield cls(base_path, artifact_name)
        else:
            for artifact_path in (base_path / "vipe").glob("*_info.pkl"):
                artifact_name = artifact_path.stem.replace("_info", "")
                yield cls(base_path, artifact_name)

    @property
    def meta_vis_path(self) -> Path:
        return self.base_path / "vipe" / f"{self.artifact_name}_vis.mp4"

    @property
    def slam_map_path(self) -> Path:
        return self.base_path / "vipe" / f"{self.artifact_name}_slam_map.pt"

    @property
    def essential_paths(self) -> list[Path]:
        return [
            self.rgb_path,
            self.pose_path,
            self.depth_path,
            self.intrinsics_path,
            self.flow_path,
            self.mask_path,
            self.mask_phrase_path,
            self.meta_info_path,
            self.meta_vis_path,
        ]

    @property
    def eval_metrics_path(self) -> Path:
        return self.base_path / "eval" / f"{self.artifact_name}_metrics.pkl"

    @property
    def eval_traj_vis_path(self) -> Path:
        return self.base_path / "eval" / f"{self.artifact_name}_trajectory_vis.png"

    @property
    def eval_gt_pose_path(self) -> Path:
        return self.base_path / "eval" / f"{self.artifact_name}_pose_gt.npz"

    @property
    def eval_gt_intrinsics_path(self) -> Path:
        return self.base_path / "eval" / f"{self.artifact_name}_intrinsics_gt.npz"

    @property
    def eval_gt_camera_type_path(self) -> Path:
        return self.base_path / "eval" / f"{self.artifact_name}_camera_gt.txt"

    @property
    def eval_gt_depth_path(self) -> Path:
        return self.base_path / "eval" / f"{self.artifact_name}_depth_gt.zip"

    @property
    def aux_vis_plot_path(self) -> Path:
        return self.base_path / "vipe_aux_vis" / f"{self.artifact_name}_plot.png"

    @property
    def aux_vis_traj_path(self) -> Path:
        return self.base_path / "vipe_aux_vis" / f"{self.artifact_name}_traj.mp4"


def write_pose_artifact(path: Path, frame_inds: np.ndarray, pose_data: np.ndarray) -> None:
    frame_inds = np.asarray(frame_inds, dtype=np.int64)
    pose_data = np.asarray(pose_data, dtype=np.float32)
    if frame_inds.ndim != 1 or pose_data.ndim != 3 or pose_data.shape[1:] != (4, 4):
        raise ValueError(
            f"Unexpected pose artifact shapes for {path}: inds={frame_inds.shape}, data={pose_data.shape}"
        )
    if frame_inds.shape[0] != pose_data.shape[0]:
        raise ValueError(
            f"Pose artifact length mismatch for {path}: inds={frame_inds.shape[0]}, data={pose_data.shape[0]}"
        )
    if frame_inds.size == 0:
        return

    order = np.argsort(frame_inds)
    path.parent.mkdir(exist_ok=True, parents=True)
    np.savez(path, data=pose_data[order], inds=frame_inds[order])


def write_intrinsics_artifact(
    intr_path: Path,
    camera_type_path: Path,
    frame_inds: np.ndarray,
    intrinsics_data: np.ndarray,
    camera_types: list[CameraType] | None = None,
) -> None:
    frame_inds = np.asarray(frame_inds, dtype=np.int64)
    intrinsics_data = np.asarray(intrinsics_data, dtype=np.float32)
    if frame_inds.ndim != 1 or intrinsics_data.ndim != 2:
        raise ValueError(
            f"Unexpected intrinsics artifact shapes for {intr_path}: inds={frame_inds.shape}, data={intrinsics_data.shape}"
        )
    if frame_inds.shape[0] != intrinsics_data.shape[0]:
        raise ValueError(
            f"Intrinsics artifact length mismatch for {intr_path}: inds={frame_inds.shape[0]}, data={intrinsics_data.shape[0]}"
        )
    if frame_inds.size == 0:
        return

    order = np.argsort(frame_inds)
    intr_path.parent.mkdir(exist_ok=True, parents=True)
    np.savez(intr_path, data=intrinsics_data[order], inds=frame_inds[order])

    if camera_types is None:
        camera_types = [CameraType.PINHOLE] * frame_inds.shape[0]
    elif len(camera_types) != frame_inds.shape[0]:
        raise ValueError(
            f"Camera type length mismatch for {camera_type_path}: inds={frame_inds.shape[0]}, camera_types={len(camera_types)}"
        )

    camera_type_path.parent.mkdir(exist_ok=True, parents=True)
    with camera_type_path.open("w") as f:
        for frame_idx, camera_type_data in zip(frame_inds[order], np.asarray(camera_types, dtype=object)[order]):
            f.write(f"{int(frame_idx)}: {camera_type_data.name}\n")


def save_camera_artifacts_from_slam_output(
    out_path: ArtifactPath,
    slam_output: "SLAMOutput",
    frame_inds: np.ndarray,
    view_idx: int = 0,
    camera_type: CameraType = CameraType.PINHOLE,
) -> None:
    trajectory = slam_output.get_view_trajectory(view_idx).matrix().cpu().numpy().astype(np.float32)
    if trajectory.shape[0] != len(frame_inds):
        raise ValueError(
            f"Trajectory/frame index mismatch for {out_path}: trajectory={trajectory.shape[0]}, frame_inds={len(frame_inds)}"
        )

    intrinsics = slam_output.intrinsics[view_idx].cpu().numpy().astype(np.float32)
    intrinsics_per_frame = np.repeat(intrinsics[None], len(frame_inds), axis=0)
    camera_types = [camera_type] * len(frame_inds)

    write_pose_artifact(out_path.pose_path, frame_inds, trajectory)
    write_intrinsics_artifact(
        out_path.intrinsics_path,
        out_path.camera_type_path,
        frame_inds,
        intrinsics_per_frame,
        camera_types,
    )


def save_pose_artifacts(out_path: ArtifactPath, cached_final_stream: VideoStream, gt: bool = False) -> None:
    # Save OpenCV cam2world matrices as 4x4 matrix in npz file
    path = out_path.eval_gt_pose_path if gt else out_path.pose_path

    pose_list = [
        (frame_data.raw_frame_idx, frame_data.pose.matrix().cpu().numpy())
        for frame_data in cached_final_stream
        if frame_data.pose is not None
    ]
    if len(pose_list) > 0:
        pose_data = np.stack([pose for _, pose in pose_list], axis=0)
        pose_inds = np.array([frame_idx for frame_idx, _ in pose_list])
        write_pose_artifact(path, pose_inds, pose_data)


def read_pose_artifacts(npz_file_path: Path) -> tuple[np.ndarray, SE3]:
    data = np.load(npz_file_path)
    return data["inds"], se3_matrix_to_se3(data["data"])


def read_pose_artifacts_benchmark(npz_file_path: Path) -> dict:
    data = np.load(npz_file_path)
    return dict(
        ids=data["inds"],
        trajectory=se3_matrix_to_se3(data["data"]),
        runtime=data.get("runtime", None),
        keyframe_ids=data.get("keyframe_ids", None),
        frame_num=len(data["inds"]),
    )


def save_intrinsics_artifacts(out_path: ArtifactPath, cached_final_stream: VideoStream, gt: bool = False) -> None:
    # Save intrinsics as [fx, fy, cx, cy] in npz file
    intr_path = out_path.eval_gt_intrinsics_path if gt else out_path.intrinsics_path
    camera_type_path = out_path.eval_gt_camera_type_path if gt else out_path.camera_type_path

    intrinsics_list = [
        (frame_data.raw_frame_idx, frame_data.intrinsics.cpu().numpy())
        for frame_data in cached_final_stream
        if frame_data.intrinsics is not None
    ]
    if len(intrinsics_list) > 0:
        intrinsics_data = np.stack([intrinsics for _, intrinsics in intrinsics_list], axis=0)
        intrinsics_inds = np.array([frame_idx for frame_idx, _ in intrinsics_list])

    camera_type_list = [
        (frame_data.raw_frame_idx, frame_data.camera_type)
        for frame_data in cached_final_stream
        if frame_data.camera_type is not None
    ]
    if len(intrinsics_list) > 0:
        camera_types = [camera_type_data for _, camera_type_data in camera_type_list]
        write_intrinsics_artifact(
            intr_path,
            camera_type_path,
            intrinsics_inds,
            intrinsics_data,
            camera_types if len(camera_types) == len(intrinsics_list) else None,
        )


def read_intrinsics_artifacts(
    intr_file_path: Path, camera_file_path: Path | None = None
) -> tuple[np.ndarray, torch.Tensor, list[CameraType]]:
    data = np.load(intr_file_path)
    inds, intrinsics = data["inds"], torch.from_numpy(data["data"])
    if camera_file_path is None or not camera_file_path.exists():
        assert intrinsics.shape[1] == 4
        camera_types = [CameraType.PINHOLE] * intrinsics.shape[0]

    else:
        with camera_file_path.open("r") as f:
            camera_types = [CameraType[line.split(":")[1].strip()] for line in f.readlines()]

    return inds, intrinsics, camera_types


def save_rgb_artifacts(out_path: ArtifactPath, cached_final_stream: VideoStream) -> None:
    # Save original RGB as H264-encoded video.
    with VideoWriter(out_path.rgb_path, cached_final_stream.fps()) as rgb_writer:
        for frame_data in cached_final_stream:
            rgb_writer.write((frame_data.rgb.cpu().numpy() * 255).astype(np.uint8))


def read_rgb_artifacts(rgb_file_path: Path) -> Iterator[tuple[int, torch.Tensor]]:
    """
    Read RGB from H264-encoded video.
    """
    reader = imageio.get_reader(rgb_file_path, "ffmpeg")
    for frame_idx, rgb in enumerate(reader):
        rgb = torch.from_numpy(rgb) / 255.0
        yield frame_idx, rgb


def save_depth_artifacts(out_path: ArtifactPath, cached_final_stream: VideoStream, gt: bool = False) -> None:
    # Save metric depth as zipped exr files.
    if gt:
        metric_depth_list = cached_final_stream.get_gt_stream_attribute(FrameAttribute.METRIC_DEPTH)
        path = out_path.eval_gt_depth_path
    else:
        metric_depth_list = cached_final_stream.get_stream_attribute(FrameAttribute.METRIC_DEPTH)
        path = out_path.depth_path

    metric_depth_list = [
        (frame_idx, depth_data.cpu().numpy())
        for frame_idx, depth_data in enumerate(metric_depth_list)
        if depth_data is not None
    ]
    if len(metric_depth_list) > 0:
        path.parent.mkdir(exist_ok=True, parents=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for frame_idx, metric_depth in metric_depth_list:
                height, width = metric_depth.shape
                header = OpenEXR.Header(width, height)
                header["channels"] = {"Z": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))}
                with tempfile.NamedTemporaryFile(suffix=".exr") as f:
                    exr = OpenEXR.OutputFile(f.name, header)
                    exr.writePixels({"Z": metric_depth.astype(np.float16).tobytes()})
                    exr.close()
                    z.write(f.name, f"{frame_idx:05d}.exr")


def read_depth_artifacts(zip_file_path: Path) -> Iterator[tuple[int, torch.Tensor]]:
    """
    Read metric depth from zipped exr files.
    """
    valid_width, valid_height = 0, 0
    with zipfile.ZipFile(zip_file_path, "r") as z:
        for file_name in sorted(z.namelist()):
            frame_idx = int(file_name.split(".")[0])
            with z.open(file_name) as f:
                try:
                    exr = OpenEXR.InputFile(f)
                except OSError:
                    # Sometimes EXR loader might fail, we return all nan maps.
                    logger.warning(f"Failed to load EXR file {zip_file_path}-{file_name}. Returning all nan maps.")
                    assert valid_width > 0 and valid_height > 0
                    yield (
                        frame_idx,
                        torch.full(
                            (valid_height, valid_width),
                            float("nan"),
                            dtype=torch.float32,
                        ),
                    )
                    continue
                header = exr.header()
                dw = header["dataWindow"]
                valid_width = width = dw.max.x - dw.min.x + 1
                valid_height = height = dw.max.y - dw.min.y + 1
                channels = exr.channels(["Z"])
                depth_data = np.frombuffer(channels[0], dtype=np.float16).reshape((height, width))
                yield frame_idx, torch.from_numpy(depth_data.copy()).float()


def read_instance_artifacts(
    zip_file_path: Path,
) -> Iterator[tuple[int, torch.Tensor]]:
    """
    Read instance mask from zipped PNG files.
    """
    with zipfile.ZipFile(zip_file_path, "r") as z:
        for file_name in sorted(z.namelist()):
            frame_idx = int(file_name.split(".")[0])
            with z.open(file_name) as f:
                mask_buffer = np.frombuffer(f.read(), dtype=np.uint8)
                mask = cv2.imdecode(mask_buffer, cv2.IMREAD_UNCHANGED)
                yield frame_idx, torch.from_numpy(mask.copy()).byte()


def read_instance_phrases(instance_phrase_path: Path) -> dict[int, str]:
    """
    Read instance phrases from txt file.
    """
    instance_phrases = {}
    with instance_phrase_path.open("r") as f:
        for line in f.readlines():
            idx, phrase = line.split(":")
            instance_phrases[int(idx)] = phrase.strip()
    return instance_phrases


def save_artifacts(out_path: ArtifactPath, cached_final_stream: VideoStream) -> None:
    """
    Save the bulky artifact set.
    """

    # Save original RGB as H264-encoded video.
    save_rgb_artifacts(out_path, cached_final_stream)

    # Save metric depth as zipped exr files.
    save_depth_artifacts(out_path, cached_final_stream)

    # Save Instance mask as zipped PNG files.
    instance_list = [
        (frame_idx, frame_data.instance)
        for frame_idx, frame_data in enumerate(cached_final_stream)
        if frame_data.instance is not None
    ]
    if len(instance_list) > 0:
        out_path.mask_path.parent.mkdir(exist_ok=True, parents=True)
        with zipfile.ZipFile(out_path.mask_path, "w", zipfile.ZIP_DEFLATED) as z:
            for frame_idx, instance in instance_list:
                _, mask_buffer = cv2.imencode(".png", instance.cpu().numpy().astype(np.uint8))
                z.writestr(f"{frame_idx:05d}.png", mask_buffer.tobytes())

    # Save Instance phrases as txt file.
    instance_phrases_combined = {}
    for frame_data in cached_final_stream:
        assert isinstance(frame_data, VideoFrame)
        if frame_data.instance_phrases is not None:
            instance_phrases_combined.update(frame_data.instance_phrases)
    if len(instance_phrases_combined) > 0:
        out_path.mask_phrase_path.parent.mkdir(exist_ok=True, parents=True)
        with out_path.mask_phrase_path.open("w") as f:
            for idx, phrase in instance_phrases_combined.items():
                f.write(f"{idx}: {phrase}\n")
