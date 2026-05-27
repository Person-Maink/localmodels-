from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Tuple

import numpy as np

from .cache import LruCache
from .loaders import (
    extract_beta_average_records,
    load_model_camera_poses,
    load_source_frames,
    load_vipe_overlay,
    source_mtime_key,
)
from .mano import load_mano_assets
from .models import FrameScene, SceneActor, VisualizationFrameResponse, VisualizationManifestResponse
from .settings import AppSettings, hand_value_to_label


def _hash_request(prefix: str, payload: dict) -> str:
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def _source_palette(source_ids: List[str], settings: AppSettings) -> Dict[str, str]:
    return {source_id: settings.source_color(index) for index, source_id in enumerate(source_ids)}


def _timeline_from_frame_sets(frame_sets: List[List[dict]]) -> List[int]:
    frame_ids = sorted({int(frame["frame_id"]) for frames in frame_sets for frame in frames})
    return frame_ids


def _center_points(points: np.ndarray, wrist_point: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float32) - np.asarray(wrist_point, dtype=np.float32).reshape(1, 3)


def _mesh_actor(actor_id: str, label: str, source_id: str, hand: str, color: str, points: np.ndarray, faces: np.ndarray, opacity: float = 0.6) -> SceneActor:
    return SceneActor(
        id=actor_id,
        kind="mesh",
        label=label,
        source_id=source_id,
        hand=hand,
        color=color,
        opacity=opacity,
        points=np.asarray(points, dtype=np.float32).tolist(),
        faces=np.asarray(faces, dtype=np.int32).tolist(),
    )


def _points_actor(actor_id: str, label: str, source_id: str, hand: str, color: str, points: np.ndarray, meta: Optional[dict] = None) -> SceneActor:
    return SceneActor(
        id=actor_id,
        kind="points",
        label=label,
        source_id=source_id,
        hand=hand,
        color=color,
        opacity=1.0,
        points=np.asarray(points, dtype=np.float32).tolist(),
        meta=meta or {},
    )


def _segments_actor(actor_id: str, label: str, source_id: Optional[str], hand: Optional[str], color: str, segments, opacity: float = 1.0, meta: Optional[dict] = None) -> SceneActor:
    return SceneActor(
        id=actor_id,
        kind="segments",
        label=label,
        source_id=source_id,
        hand=hand,
        color=color,
        opacity=opacity,
        segments=np.asarray(segments, dtype=np.float32).tolist(),
        meta=meta or {},
    )


def _make_hand_segments(points: np.ndarray) -> List[List[List[float]]]:
    valid = [(i, j) for i, j in [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
    ] if i < len(points) and j < len(points)]
    return [[points[i].tolist(), points[j].tolist()] for i, j in valid]


def _make_camera_frustum(pose_wc: np.ndarray, fov_deg: float, aspect: float, scale: float) -> List[List[List[float]]]:
    center = pose_wc[:3, 3]
    fov = np.deg2rad(float(fov_deg))
    h = np.tan(fov / 2.0)
    w = h * float(aspect)
    corners_cam = np.array([[-w, -h, 1.0], [w, -h, 1.0], [w, h, 1.0], [-w, h, 1.0]], dtype=np.float32) * float(scale)
    rotation = pose_wc[:3, :3]
    translation = pose_wc[:3, 3]
    corners_world = (rotation @ corners_cam.T).T + translation
    segments = []
    for corner in corners_world:
        segments.append([center.tolist(), corner.tolist()])
    for index in range(4):
        segments.append([corners_world[index].tolist(), corners_world[(index + 1) % 4].tolist()])
    return segments


def _trajectory_segments(points: np.ndarray) -> List[List[List[float]]]:
    return [[points[index].tolist(), points[index + 1].tolist()] for index in range(len(points) - 1)]


class VisualizationService:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.manifest_cache: LruCache[str, dict] = LruCache(max_items=settings.max_cache_items)
        self.frame_cache: LruCache[str, dict] = LruCache(max_items=settings.max_cache_items)

    def build_manifest(self, mode_id: str, sources: List[dict], overlays: List[dict], params: dict) -> VisualizationManifestResponse:
        cache_payload = {
            "mode_id": mode_id,
            "source_keys": [source_mtime_key(source["path"]) for source in sources],
            "overlay_paths": [overlay["pose_path"] for overlay in overlays],
            "params": params,
        }
        request_id = _hash_request("viz", cache_payload)
        cached = self.manifest_cache.get(request_id)
        if cached is not None:
            return VisualizationManifestResponse(**cached)

        frame_sets = []
        if mode_id == "beta_average_meshes":
            for source in sources:
                frame_sets.append(extract_beta_average_records(source, self.settings)["frames"])
        else:
            include_camera_space = mode_id == "camera_space_meshes"
            wrist_joint_idx = int(params.get("wrist_joint_idx", self.settings.default_wrist_joint_index))
            for source in sources:
                frame_sets.append(
                    load_source_frames(
                        source,
                        settings=self.settings,
                        include_camera_space=include_camera_space,
                        wrist_joint_idx=wrist_joint_idx,
                    )
                )

        frame_ids = _timeline_from_frame_sets(frame_sets)
        source_colors = _source_palette([source["id"] for source in sources], self.settings)
        payload = {
            "request_id": request_id,
            "mode_id": mode_id,
            "fps": float(params.get("fps", self.settings.default_fps)),
            "frame_ids": frame_ids,
            "source_colors": source_colors,
            "available_hands": ["left", "right", "unknown"],
            "static_actors": self._static_actors(mode_id, sources, overlays, source_colors, params),
            "camera_display": self._camera_settings(params),
        }
        self.manifest_cache.set(request_id, payload)
        return VisualizationManifestResponse(**payload)

    def _camera_settings(self, params: dict) -> dict:
        base = self.settings.camera_display.to_dict()
        for key in base:
            if key in params:
                base[key] = params[key]
        return base

    def _static_actors(self, mode_id: str, sources: List[dict], overlays: List[dict], source_colors: Dict[str, str], params: dict) -> List[SceneActor]:
        if mode_id == "camera_trajectories":
            camera_settings = self._camera_settings(params)
            actors: List[SceneActor] = []
            for source in sources:
                payload = load_model_camera_poses(
                    source,
                    self.settings,
                    invert_cam_t=bool(camera_settings["invert_cam_t"]),
                )
                poses = np.asarray(payload["poses_wc"], dtype=np.float32)
                if camera_settings["center_to_first_frame"] and len(poses) > 0:
                    shift = poses[0, :3, 3].copy()
                    poses[:, :3, 3] = (poses[:, :3, 3] - shift.reshape(1, 3)) * float(camera_settings["translation_scale"])
                rights = np.asarray(payload["right"], dtype=np.int32)
                for hand_value in (1, 0, -1):
                    indices = np.where(rights == hand_value)[0]
                    if len(indices) < 2:
                        continue
                    points = poses[indices, :3, 3]
                    actors.append(
                        _segments_actor(
                            actor_id=f"track:{source['id']}:{hand_value}",
                            label=f"{source['label']} {hand_value_to_label(hand_value)} track",
                            source_id=source["id"],
                            hand=hand_value_to_label(hand_value),
                            color=source_colors[source["id"]],
                            segments=_trajectory_segments(points),
                            meta={"static": True},
                        )
                    )
            for overlay in overlays:
                payload = load_vipe_overlay(overlay)
                poses = np.asarray(payload["poses_wc"], dtype=np.float32)
                if camera_settings["center_to_first_frame"] and len(poses) > 0:
                    shift = poses[0, :3, 3].copy()
                    poses[:, :3, 3] = (poses[:, :3, 3] - shift.reshape(1, 3)) * float(camera_settings["translation_scale"])
                if len(poses) > 1:
                    actors.append(
                        _segments_actor(
                            actor_id=f"track:{overlay['id']}",
                            label="ViPE overlay track",
                            source_id=None,
                            hand=None,
                            color="#1f7a53",
                            segments=_trajectory_segments(poses[:, :3, 3]),
                            meta={"static": True, "overlay": True},
                        )
                    )
            return actors

        if mode_id == "bounding_boxes":
            actors = []
            time_spacing = float(params.get("time_spacing", 10.0))
            for source in sources:
                frames = load_source_frames(source, settings=self.settings)
                for hand_value in (1, 0, -1):
                    centers = []
                    for frame in frames:
                        for hand in frame["hands"]:
                            if hand["right"] != hand_value:
                                continue
                            if hand.get("box_center") is None or hand.get("box_size") is None:
                                continue
                            box_center = np.asarray(hand["box_center"], dtype=np.float32)
                            centers.append([box_center[0], -box_center[1], frame["frame_id"] * time_spacing])
                    if len(centers) > 1:
                        actors.append(
                            _segments_actor(
                                actor_id=f"bbox-track:{source['id']}:{hand_value}",
                                label=f"{source['label']} {hand_value_to_label(hand_value)} centers",
                                source_id=source["id"],
                                hand=hand_value_to_label(hand_value),
                                color=source_colors[source["id"]],
                                segments=_trajectory_segments(np.asarray(centers, dtype=np.float32)),
                                meta={"static": True},
                            )
                        )
            return actors
        return []

    def build_frames(self, mode_id: str, sources: List[dict], overlays: List[dict], params: dict, frame_start: int, frame_end: int) -> VisualizationFrameResponse:
        cache_payload = {
            "mode_id": mode_id,
            "source_keys": [source_mtime_key(source["path"]) for source in sources],
            "overlay_paths": [overlay["pose_path"] for overlay in overlays],
            "params": params,
            "frame_start": frame_start,
            "frame_end": frame_end,
        }
        request_id = _hash_request("frames", cache_payload)
        cached = self.frame_cache.get(request_id)
        if cached is not None:
            return VisualizationFrameResponse(**cached)

        if mode_id == "beta_average_meshes":
            bundles = {source["id"]: extract_beta_average_records(source, self.settings) for source in sources}
            frames_by_source = {source_id: bundle["frames"] for source_id, bundle in bundles.items()}
            faces_right = {source_id: bundle["mano"]["faces_right"] for source_id, bundle in bundles.items()}
            faces_left = {source_id: bundle["mano"]["faces_left"] for source_id, bundle in bundles.items()}
        else:
            wrist_joint_idx = int(params.get("wrist_joint_idx", self.settings.default_wrist_joint_index))
            frames_by_source = {
                source["id"]: load_source_frames(
                    source,
                    settings=self.settings,
                    include_camera_space=(mode_id == "camera_space_meshes"),
                    wrist_joint_idx=wrist_joint_idx,
                )
                for source in sources
            }
            mano = load_mano_assets(str(self.settings.mano_right_path))
            faces_right = {source["id"]: mano["faces_right"] for source in sources if source["family"] != "mediapipe"}
            faces_left = {source["id"]: mano["faces_left"] for source in sources if source["family"] != "mediapipe"}

        all_frame_ids = sorted(
            {
                frame["frame_id"]
                for frames in frames_by_source.values()
                for frame in frames
                if int(frame_start) <= int(frame["frame_id"]) <= int(frame_end)
            }
        )
        source_colors = _source_palette([source["id"] for source in sources], self.settings)
        scenes = []
        for frame_id in all_frame_ids:
            actors: List[SceneActor] = []
            for source in sources:
                source_frames = frames_by_source[source["id"]]
                frame = next((item for item in source_frames if int(item["frame_id"]) == int(frame_id)), None)
                if frame is None:
                    continue
                color = source_colors[source["id"]]
                actors.extend(
                    self._actors_for_frame(
                        mode_id=mode_id,
                        source=source,
                        frame=frame,
                        faces_right=faces_right.get(source["id"]),
                        faces_left=faces_left.get(source["id"]),
                        color=color,
                        params=params,
                    )
                )
            actors.extend(self._overlay_actors(mode_id, overlays, frame_id, params))
            scenes.append(FrameScene(frame_id=int(frame_id), actors=actors))

        payload = {"request_id": request_id, "mode_id": mode_id, "scenes": [scene.model_dump() for scene in scenes]}
        self.frame_cache.set(request_id, payload)
        return VisualizationFrameResponse(**payload)

    def _actors_for_frame(self, mode_id: str, source: dict, frame: dict, faces_right, faces_left, color: str, params: dict) -> List[SceneActor]:
        actors = []
        wrist_joint_idx = int(params.get("wrist_joint_idx", self.settings.default_wrist_joint_index))
        for hand_index, hand in enumerate(frame["hands"]):
            hand_label = hand_value_to_label(hand["right"])
            actor_id = f"{source['id']}:{frame['frame_id']}:{hand_index}"
            if source["family"] == "mediapipe":
                points = np.asarray(hand["points"], dtype=np.float32)
                if mode_id == "wrist_grounded" and len(points) > wrist_joint_idx:
                    points = _center_points(points, points[wrist_joint_idx])
                actors.append(
                    _points_actor(
                        actor_id=actor_id,
                        label=f"{source['label']} {hand_label}",
                        source_id=source["id"],
                        hand=hand_label,
                        color=color,
                        points=points,
                    )
                )
                segments = _make_hand_segments(points)
                if segments:
                    actors.append(
                        _segments_actor(
                            actor_id=f"{actor_id}:segments",
                            label=f"{source['label']} {hand_label} skeleton",
                            source_id=source["id"],
                            hand=hand_label,
                            color=color,
                            segments=segments,
                        )
                    )
                continue

            if mode_id == "camera_space_meshes":
                verts = np.asarray(hand["verts_camera_space"], dtype=np.float32)
            else:
                verts = np.asarray(hand["verts_world"], dtype=np.float32)
                if mode_id == "wrist_grounded":
                    joints = np.asarray(hand["joints_world"], dtype=np.float32)
                    if len(joints) > wrist_joint_idx:
                        verts = _center_points(verts, joints[wrist_joint_idx])
            faces = faces_right if int(hand["right"]) == 1 else faces_left
            actors.append(
                _mesh_actor(
                    actor_id=actor_id,
                    label=f"{source['label']} {hand_label}",
                    source_id=source["id"],
                    hand=hand_label,
                    color=color,
                    points=verts,
                    faces=faces,
                    opacity=0.62,
                )
            )

        if mode_id == "bounding_boxes":
            box_stride = max(1, int(params.get("box_stride", 5)))
            line_width = float(params.get("line_width", 2.0))
            time_spacing = float(params.get("time_spacing", 10.0))
            center_radius = float(params.get("center_radius", 4.0))
            for hand_index, hand in enumerate(frame["hands"]):
                if hand.get("box_center") is None or hand.get("box_size") is None:
                    continue
                cx, cy = np.asarray(hand["box_center"], dtype=np.float32).reshape(2)
                half = float(hand["box_size"]) / 2.0
                depth = float(frame["frame_id"]) * time_spacing
                corners = np.array(
                    [
                        [cx - half, -(cy - half), depth],
                        [cx + half, -(cy - half), depth],
                        [cx + half, -(cy + half), depth],
                        [cx - half, -(cy + half), depth],
                    ],
                    dtype=np.float32,
                )
                segments = [[corners[idx].tolist(), corners[(idx + 1) % 4].tolist()] for idx in range(4)]
                hand_label = hand_value_to_label(hand["right"])
                actors.append(
                    _segments_actor(
                        actor_id=f"{source['id']}:{frame['frame_id']}:{hand_index}:bbox",
                        label=f"{source['label']} {hand_label} bbox",
                        source_id=source["id"],
                        hand=hand_label,
                        color=color,
                        segments=segments,
                        meta={"line_width": line_width, "box_stride": box_stride},
                    )
                )
                actors.append(
                    _points_actor(
                        actor_id=f"{source['id']}:{frame['frame_id']}:{hand_index}:center",
                        label=f"{source['label']} {hand_label} center",
                        source_id=source["id"],
                        hand=hand_label,
                        color=color,
                        points=np.asarray([[cx, -cy, depth]], dtype=np.float32),
                        meta={"point_radius": center_radius},
                    )
                )
        if mode_id == "camera_trajectories":
            pose_payload = load_model_camera_poses(
                source,
                self.settings,
                invert_cam_t=bool(self._camera_settings(params)["invert_cam_t"]),
            )
            camera_settings = self._camera_settings(params)
            indices = np.where(np.asarray(pose_payload["frame_id"], dtype=np.int32) == int(frame["frame_id"]))[0]
            poses = np.asarray(pose_payload["poses_wc"], dtype=np.float32)
            rights = np.asarray(pose_payload["right"], dtype=np.int32)
            if camera_settings["center_to_first_frame"] and len(poses) > 0:
                shift = poses[0, :3, 3].copy()
                poses[:, :3, 3] = (poses[:, :3, 3] - shift.reshape(1, 3)) * float(camera_settings["translation_scale"])
            for position, index in enumerate(indices):
                hand_label = hand_value_to_label(rights[index])
                frustum = _make_camera_frustum(
                    poses[index],
                    fov_deg=float(camera_settings["fov_deg"]),
                    aspect=float(camera_settings["aspect"]),
                    scale=float(camera_settings["frustum_scale"]),
                )
                actors.append(
                    _segments_actor(
                        actor_id=f"{source['id']}:{frame['frame_id']}:{position}:frustum",
                        label=f"{source['label']} {hand_label} frustum",
                        source_id=source["id"],
                        hand=hand_label,
                        color=color,
                        segments=frustum,
                    )
                )
        return actors

    def _overlay_actors(self, mode_id: str, overlays: List[dict], frame_id: int, params: dict) -> List[SceneActor]:
        if mode_id != "camera_trajectories":
            return []
        actors = []
        camera_settings = self._camera_settings(params)
        for overlay in overlays:
            payload = load_vipe_overlay(overlay)
            poses = np.asarray(payload["poses_wc"], dtype=np.float32)
            frame_ids = np.asarray(payload["frame_id"], dtype=np.int32)
            if camera_settings["center_to_first_frame"] and len(poses) > 0:
                shift = poses[0, :3, 3].copy()
                poses[:, :3, 3] = (poses[:, :3, 3] - shift.reshape(1, 3)) * float(camera_settings["translation_scale"])
            indices = np.where(frame_ids == int(frame_id))[0]
            for position, index in enumerate(indices):
                frustum = _make_camera_frustum(
                    poses[index],
                    fov_deg=float(camera_settings["fov_deg"]),
                    aspect=float(camera_settings["aspect"]),
                    scale=float(camera_settings["frustum_scale"]),
                )
                actors.append(
                    _segments_actor(
                        actor_id=f"{overlay['id']}:{frame_id}:{position}",
                        label="ViPE overlay frustum",
                        source_id=None,
                        hand=None,
                        color="#1f7a53",
                        segments=frustum,
                        meta={"overlay": True},
                    )
                )
        return actors
