from __future__ import annotations

import json
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from torch.utils.data import DataLoader

from finetune_wilor_common import (
    DetectedVideoHandDataset,
    ViPECameraIndex,
    build_detection_samples,
    discover_videos,
    filter_videos_with_vipe_artifacts,
    split_samples_by_frame,
)
from frame_store import FrameRecord, FrameStore
from temporal_losses import (
    TemporalWindowDataset,
    build_temporal_windows,
    temporal_window_collate,
)
from wilor.datasets.vitdet_dataset import ViTDetDataset


EMPTY_WINDOW_STATS = {
    "window_count": 0,
    "dropped_window_count": 0,
    "stream_count": 0,
    "broken_segment_count": 0,
}


@dataclass(frozen=True)
class WindowDataBundle:
    video_names: list[str]
    split_stats: dict[str, Any]
    train_window_stats: dict[str, Any]
    val_window_stats: dict[str, Any]
    train_dataloader: Iterable[dict[str, Any]]
    val_dataloader: Iterable[dict[str, Any]] | None
    window_mode: str = "eager_detection"
    sample_count_mode: str = "detected_samples"


def resolve_video_names(
    args,
    log_fn: Callable[[str], None],
) -> list[str]:
    if args.all_videos:
        log_fn("Discovering videos from image folder because --all_videos is enabled.")
        video_names = discover_videos(
            args.image_folder,
            frame_cache_root=getattr(args, "frame_cache_root", None),
        )
    else:
        video_names = sorted(set(args.videos))
        log_fn(
            "Using explicitly requested video selection: "
            f"{', '.join(video_names) if video_names else '<none>'}"
        )
    if not video_names:
        raise ValueError("No videos selected. Pass --video ... or use --all_videos.")

    log_fn(f"Filtering {len(video_names)} selected video(s) for matching ViPE artifacts.")
    valid_video_names, missing_vipe_artifacts = filter_videos_with_vipe_artifacts(
        video_names,
        args.pose_dir,
        args.intrinsics_dir,
    )
    if missing_vipe_artifacts:
        if args.all_videos:
            print(
                f"Skipping {len(missing_vipe_artifacts)} discovered video(s) without matching "
                "ViPE pose/intrinsics artifacts:"
            )
            for video_name, missing_paths in sorted(missing_vipe_artifacts.items()):
                print(f"  - {video_name}: missing {', '.join(missing_paths)}")
            video_names = valid_video_names
        else:
            missing_lines = [
                f"{video_name}: missing {', '.join(missing_paths)}"
                for video_name, missing_paths in sorted(missing_vipe_artifacts.items())
            ]
            raise ValueError(
                "The following requested video(s) do not have matching ViPE pose/intrinsics artifacts:\n"
                + "\n".join(missing_lines)
            )

    if not video_names:
        raise ValueError(
            "No videos remain after filtering for matching ViPE pose/intrinsics artifacts."
        )

    log_fn("Proceeding with video(s): " + ", ".join(video_names))
    return video_names


def _resolve_frame_store(args) -> FrameStore:
    cache_root = getattr(args, "frame_cache_root", None) or None
    return FrameStore(args.image_folder, cache_root=cache_root)


def _resolve_lazy_video_names(
    args,
    frame_store: FrameStore,
    log_fn: Callable[[str], None],
) -> list[str]:
    if args.all_videos:
        log_fn("Discovering videos from FrameStore because lazy detection is enabled.")
        video_names = sorted(
            {
                video_name
                for video_name in (
                    frame_store.list_videos() + frame_store.list_unavailable_videos()
                )
                if video_name != "single_images"
            }
        )
    else:
        video_names = sorted(set(args.videos))
        log_fn(
            "Using explicitly requested video selection: "
            f"{', '.join(video_names) if video_names else '<none>'}"
        )
    if not video_names:
        raise ValueError("No videos selected. Pass --video ... or use --all_videos.")

    log_fn(f"Filtering {len(video_names)} selected video(s) for matching ViPE artifacts.")
    valid_video_names, missing_vipe_artifacts = filter_videos_with_vipe_artifacts(
        video_names,
        args.pose_dir,
        args.intrinsics_dir,
    )
    if missing_vipe_artifacts:
        if args.all_videos:
            print(
                f"Skipping {len(missing_vipe_artifacts)} discovered video(s) without matching "
                "ViPE pose/intrinsics artifacts:"
            )
            for video_name, missing_paths in sorted(missing_vipe_artifacts.items()):
                print(f"  - {video_name}: missing {', '.join(missing_paths)}")
            video_names = valid_video_names
        else:
            missing_lines = [
                f"{video_name}: missing {', '.join(missing_paths)}"
                for video_name, missing_paths in sorted(missing_vipe_artifacts.items())
            ]
            raise ValueError(
                "The following requested video(s) do not have matching ViPE pose/intrinsics artifacts:\n"
                + "\n".join(missing_lines)
            )

    available_video_names: list[str] = []
    unavailable_messages: dict[str, str] = {}
    for video_name in video_names:
        if frame_store.has_video(video_name):
            available_video_names.append(video_name)
            continue
        unavailable_messages[video_name] = (
            frame_store.explain_unavailable(video_name)
            or f"No lazy frame source was found for video '{video_name}'."
        )

    if unavailable_messages:
        if args.all_videos:
            print(
                f"Skipping {len(unavailable_messages)} discovered video(s) without a usable "
                "ZIP sidecar cache or legacy *_frames directory:"
            )
            for video_name, reason in sorted(unavailable_messages.items()):
                print(f"  - {video_name}: {reason}")
            video_names = available_video_names
        else:
            missing_lines = [
                f"{video_name}: {reason}"
                for video_name, reason in sorted(unavailable_messages.items())
            ]
            raise ValueError(
                "The following requested video(s) do not have a usable lazy frame source:\n"
                + "\n".join(missing_lines)
            )

    if not video_names:
        raise ValueError(
            "No videos remain after filtering for matching ViPE artifacts and usable frame sources."
        )

    log_fn("Proceeding with video(s): " + ", ".join(video_names))
    return video_names


def _collect_frame_records(
    frame_store: FrameStore,
    video_names: Sequence[str],
    sample_limit: int,
) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    for video_name in sorted(set(video_names)):
        records.extend(frame_store.iter_video_frames(video_name))
    records.sort(key=lambda record: (record.video_name, int(record.frame_id)))
    if sample_limit > 0:
        records = records[:sample_limit]
    return records


def _split_frame_records(
    records: Sequence[FrameRecord],
    validation_split: float,
    seed: int,
) -> tuple[list[FrameRecord], list[FrameRecord], dict[str, Any]]:
    if not 0.0 <= validation_split < 1.0:
        raise ValueError(
            f"validation_split must be in [0.0, 1.0). Received {validation_split}."
        )

    if not records:
        return [], [], {
            "validation_split": validation_split,
            "total_frames": 0,
            "train_frames": 0,
            "val_frames": 0,
            "total_samples": 0,
            "train_samples": 0,
            "val_samples": 0,
        }

    unique_frame_keys = sorted(
        {(record.video_name, int(record.frame_id)) for record in records},
        key=lambda item: (item[0], item[1]),
    )
    val_frame_count = int(round(len(unique_frame_keys) * validation_split))
    if validation_split > 0.0 and len(unique_frame_keys) > 1:
        val_frame_count = max(1, val_frame_count)
        val_frame_count = min(val_frame_count, len(unique_frame_keys) - 1)
    else:
        val_frame_count = 0

    shuffled_frame_keys = list(unique_frame_keys)
    random.Random(seed).shuffle(shuffled_frame_keys)
    val_frame_keys = set(shuffled_frame_keys[:val_frame_count])

    train_records: list[FrameRecord] = []
    val_records: list[FrameRecord] = []
    for record in records:
        target = (
            val_records
            if (record.video_name, int(record.frame_id)) in val_frame_keys
            else train_records
        )
        target.append(record)

    return train_records, val_records, {
        "validation_split": validation_split,
        "total_frames": len(unique_frame_keys),
        "train_frames": len(unique_frame_keys) - len(val_frame_keys),
        "val_frames": len(val_frame_keys),
        "total_samples": 0,
        "train_samples": 0,
        "val_samples": 0,
    }


def _frame_segment_to_windows(
    segment: list[FrameRecord],
    window_size: int,
    window_stride: int,
) -> tuple[list[list[FrameRecord]], int]:
    if len(segment) < window_size:
        return [], 1

    windows = [
        segment[start : start + window_size]
        for start in range(0, len(segment) - window_size + 1, window_stride)
    ]
    last_start = len(segment) - window_size
    has_remainder = last_start % window_stride != 0
    dropped_window_count = 1 if has_remainder else 0
    return windows, dropped_window_count


def _build_frame_windows(
    frame_records: Sequence[FrameRecord],
    window_size: int,
    window_stride: int,
    max_frame_gap: int,
) -> tuple[list[list[FrameRecord]], dict[str, Any]]:
    if window_size < 3:
        raise ValueError(f"window_size must be >= 3. Received {window_size}.")
    if window_stride < 1:
        raise ValueError(f"window_stride must be >= 1. Received {window_stride}.")
    if max_frame_gap < 1:
        raise ValueError(f"max_frame_gap must be >= 1. Received {max_frame_gap}.")

    records_by_video: dict[str, list[FrameRecord]] = defaultdict(list)
    for record in frame_records:
        records_by_video[record.video_name].append(record)

    all_windows: list[list[FrameRecord]] = []
    dropped_window_count = 0
    broken_segment_count = 0

    for video_name in sorted(records_by_video):
        sorted_records = sorted(
            records_by_video[video_name],
            key=lambda record: int(record.frame_id),
        )
        current_segment: list[FrameRecord] = []
        prev_frame_idx: int | None = None

        for record in sorted_records:
            frame_idx = int(record.frame_id)
            if prev_frame_idx is None or frame_idx - prev_frame_idx <= max_frame_gap:
                current_segment.append(record)
            else:
                segment_windows, dropped = _frame_segment_to_windows(
                    current_segment,
                    window_size,
                    window_stride,
                )
                all_windows.extend(segment_windows)
                dropped_window_count += dropped
                broken_segment_count += 1
                current_segment = [record]
            prev_frame_idx = frame_idx

        if current_segment:
            segment_windows, dropped = _frame_segment_to_windows(
                current_segment,
                window_size,
                window_stride,
            )
            all_windows.extend(segment_windows)
            dropped_window_count += dropped

    return all_windows, {
        "stream_count": len(records_by_video),
        "window_count": len(all_windows),
        "dropped_window_count": dropped_window_count,
        "broken_segment_count": broken_segment_count,
    }


class OnDemandDetectionCache:
    def __init__(
        self,
        frame_store: FrameStore,
        detector,
        camera_index: ViPECameraIndex,
        video_names: Sequence[str],
        detection_conf: float,
        detection_cache_path: str | None,
        log_fn: Callable[[str], None],
    ):
        self.frame_store = frame_store
        self.detector = detector
        self.camera_index = camera_index
        self.video_name_to_idx = {
            video_name: idx for idx, video_name in enumerate(sorted(set(video_names)))
        }
        self.detection_conf = float(detection_conf)
        self.detection_cache_path = (
            Path(detection_cache_path).expanduser().resolve()
            if detection_cache_path
            else None
        )
        self.log_fn = log_fn
        self._frame_samples: dict[tuple[str, int], list[dict[str, Any]]] = {}
        self._dirty = False
        self._new_frame_count = 0
        self._loaded_frame_count = 0
        self._load_existing_cache()

    def _load_existing_cache(self) -> None:
        if self.detection_cache_path is None or not self.detection_cache_path.exists():
            return

        with open(self.detection_cache_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        selected_videos = set(self.video_name_to_idx)
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for sample in payload:
            video_name = str(sample.get("video_name", ""))
            if video_name not in selected_videos:
                continue
            frame_idx = int(sample["frame_idx"])
            normalized = dict(sample)
            normalized["video_name"] = video_name
            normalized["video_idx"] = self.video_name_to_idx[video_name]
            normalized["frame_idx"] = frame_idx
            normalized["det_idx"] = int(sample["det_idx"])
            normalized["right"] = float(sample["right"])
            normalized["bbox"] = [float(value) for value in sample["bbox"]]
            grouped[(video_name, frame_idx)].append(normalized)

        for frame_key, frame_samples in grouped.items():
            frame_samples.sort(key=lambda item: int(item["det_idx"]))
            self._frame_samples[frame_key] = frame_samples

        self._loaded_frame_count = len(self._frame_samples)
        self.log_fn(
            "Loaded lazy detection cache with "
            f"{self._loaded_frame_count} frame(s) from {self.detection_cache_path}"
        )

    def _frame_display_path(self, record: FrameRecord) -> tuple[str, str]:
        if record.source_kind in {"loose_frames", "single_image"}:
            path = Path(record.source_key)
            return str(path), str(path)

        source_path = self.frame_store.get_source_path(record.video_name)
        source_display = str(source_path) if source_path is not None else record.video_name
        synthetic_name = f"{record.video_name}/{record.frame_name}.jpg"
        return f"{source_display}::{record.source_key}", synthetic_name

    def get_cached_frame_samples(
        self,
        video_name: str,
        frame_idx: int,
    ) -> list[dict[str, Any]]:
        frame_key = (str(video_name), int(frame_idx))
        if frame_key not in self._frame_samples:
            raise KeyError(
                f"Frame detections have not been materialized yet for {video_name} frame {frame_idx}."
            )
        return self._frame_samples[frame_key]

    def get_frame_samples(self, record: FrameRecord) -> list[dict[str, Any]]:
        frame_key = (record.video_name, int(record.frame_id))
        cached = self._frame_samples.get(frame_key)
        if cached is not None:
            return cached

        img_cv2 = self.frame_store.get_frame(record.video_name, int(record.frame_id))
        if img_cv2 is None:
            self._frame_samples[frame_key] = []
            self._dirty = True
            return []

        camera_target = self.camera_index.resolve(record.video_name, int(record.frame_id))
        detections = self.detector(img_cv2, conf=self.detection_conf, verbose=False)[0]
        img_path, img_path_rel = self._frame_display_path(record)
        frame_samples: list[dict[str, Any]] = []
        for det in detections:
            bbox = det.boxes.data.cpu().detach().reshape(-1, det.boxes.data.shape[-1])[0].numpy()
            right = float(det.boxes.cls.cpu().detach().reshape(-1)[0].item())
            x_center = float((bbox[0] + bbox[2]) * 0.5)
            frame_samples.append(
                {
                    "img_path": img_path,
                    "img_path_rel": img_path_rel,
                    "video_name": record.video_name,
                    "video_idx": self.video_name_to_idx[record.video_name],
                    "frame_idx": int(record.frame_id),
                    "frame_name": record.frame_name,
                    "bbox": [float(value) for value in bbox[:4]],
                    "right": right,
                    "x_center": x_center,
                    **{
                        key: value.tolist() if isinstance(value, np.ndarray) else value.item()
                        if isinstance(value, np.generic)
                        else value
                        for key, value in camera_target.items()
                    },
                }
            )

        frame_samples.sort(key=lambda item: (item["right"], item["x_center"]))
        for det_idx, sample in enumerate(frame_samples):
            sample["det_idx"] = det_idx

        self._frame_samples[frame_key] = frame_samples
        self._dirty = True
        self._new_frame_count += 1
        if self._new_frame_count == 1 or self._new_frame_count % 25 == 0:
            self.log_fn(
                "Lazy detector has materialized "
                f"{self._new_frame_count} new frame(s) "
                f"({self._loaded_frame_count} restored from cache)."
            )
        if self._new_frame_count % 50 == 0:
            self.flush()
        return frame_samples

    def flush(self) -> None:
        if not self._dirty or self.detection_cache_path is None:
            return

        serializable_samples: list[dict[str, Any]] = []
        for frame_key in sorted(self._frame_samples):
            frame_samples = self._frame_samples[frame_key]
            for sample in frame_samples:
                serializable_samples.append(dict(sample))

        self.detection_cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.detection_cache_path.with_suffix(
            self.detection_cache_path.suffix + ".tmp"
        )
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(serializable_samples, handle)
        tmp_path.replace(self.detection_cache_path)
        self._dirty = False
        self.log_fn(
            "Flushed lazy detection cache with "
            f"{len(self._frame_samples)} frame(s) to {self.detection_cache_path}"
        )


class LazyCropMaterializer:
    def __init__(
        self,
        model_cfg,
        frame_store: FrameStore,
        detection_cache: OnDemandDetectionCache,
        rescale_factor: float = 2.0,
        include_path_metadata: bool = False,
        max_cached_frames: int = 64,
    ):
        self.model_cfg = model_cfg
        self.frame_store = frame_store
        self.detection_cache = detection_cache
        self.rescale_factor = rescale_factor
        self.include_path_metadata = include_path_metadata
        self.max_cached_frames = max(0, int(max_cached_frames))
        self._frame_item_cache: OrderedDict[
            tuple[str, int], list[dict[str, Any]]
        ] = OrderedDict()

    def _get_frame_items(
        self,
        video_name: str,
        frame_idx: int,
    ) -> list[dict[str, Any]]:
        frame_key = (str(video_name), int(frame_idx))
        cached_items = self._frame_item_cache.get(frame_key)
        if cached_items is not None:
            self._frame_item_cache.move_to_end(frame_key)
            return cached_items

        img_cv2 = self.frame_store.get_frame(video_name, frame_idx)
        if img_cv2 is None:
            raise RuntimeError(f"Failed to decode frame {frame_idx} from video '{video_name}'.")

        frame_samples = self.detection_cache.get_cached_frame_samples(video_name, frame_idx)
        crop_dataset = ViTDetDataset(
            self.model_cfg,
            img_cv2,
            np.asarray([sample["bbox"] for sample in frame_samples], dtype=np.float32),
            np.asarray([sample["right"] for sample in frame_samples], dtype=np.float32),
            rescale_factor=self.rescale_factor,
        )
        cached_items = [dict(crop_dataset[item_idx]) for item_idx in range(len(frame_samples))]
        if self.max_cached_frames != 0:
            self._frame_item_cache[frame_key] = cached_items
            self._frame_item_cache.move_to_end(frame_key)
            while len(self._frame_item_cache) > self.max_cached_frames:
                self._frame_item_cache.popitem(last=False)
        return cached_items

    def materialize_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        frame_items = self._get_frame_items(
            str(sample["video_name"]),
            int(sample["frame_idx"]),
        )
        frame_item_index = int(sample["det_idx"])
        item = dict(frame_items[frame_item_index])
        if self.include_path_metadata:
            item["imgname"] = sample["img_path"]
            item["imgname_rel"] = sample["img_path_rel"]
            item["video_name"] = sample["video_name"]
        item["video_idx"] = np.int64(sample["video_idx"])
        item["frame_idx"] = np.int64(sample["frame_idx"])
        item["det_idx"] = np.int64(sample["det_idx"])
        item["camera_target_t_full"] = np.asarray(sample["camera_target_t_full"], dtype=np.float32)
        item["camera_target_valid"] = np.float32(sample["camera_target_valid"])
        item["camera_target_pose_frame_idx"] = np.int64(sample["camera_target_pose_frame_idx"])
        item["camera_target_intrinsics_frame_idx"] = np.int64(
            sample["camera_target_intrinsics_frame_idx"]
        )
        item["camera_target_focal"] = np.asarray(sample["camera_target_focal"], dtype=np.float32)
        item["camera_target_center"] = np.asarray(sample["camera_target_center"], dtype=np.float32)
        item["camera_target_used_fallback"] = np.float32(sample["camera_target_used_fallback"])
        return item


class LazyTemporalBatchIterable:
    def __init__(
        self,
        split_name: str,
        frame_windows: Sequence[Sequence[FrameRecord]],
        batch_size: int,
        detection_cache: OnDemandDetectionCache,
        crop_materializer: LazyCropMaterializer,
        shuffle: bool,
        seed: int,
        log_fn: Callable[[str], None],
    ):
        self.split_name = split_name
        self.frame_windows = [list(window) for window in frame_windows]
        self.batch_size = int(batch_size)
        self.detection_cache = detection_cache
        self.crop_materializer = crop_materializer
        self.shuffle = shuffle
        self.seed = int(seed)
        self.log_fn = log_fn
        self._iteration = 0

    def _expand_frame_window(
        self,
        frame_window: Sequence[FrameRecord],
    ) -> list[list[dict[str, Any]]]:
        stream_samples: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)

        for record in frame_window:
            frame_samples = self.detection_cache.get_frame_samples(record)
            if not frame_samples:
                return []
            for sample in frame_samples:
                stream_key = (
                    int(round(float(sample["right"]))),
                    int(sample["det_idx"]),
                )
                stream_samples[stream_key].append(sample)

        complete_windows = [
            samples
            for stream_key, samples in sorted(stream_samples.items())
            if len(samples) == len(frame_window)
        ]
        return complete_windows

    def __iter__(self):
        iteration_index = self._iteration
        self._iteration += 1
        ordered_windows = list(self.frame_windows)
        if self.shuffle:
            random.Random(self.seed + iteration_index).shuffle(ordered_windows)

        yielded_batches = 0
        materialized_windows = 0
        batch_windows: list[list[dict[str, Any]]] = []

        try:
            for frame_window in ordered_windows:
                for sample_window in self._expand_frame_window(frame_window):
                    batch_windows.append(
                        [
                            self.crop_materializer.materialize_sample(sample)
                            for sample in sample_window
                        ]
                    )
                    materialized_windows += 1
                    if len(batch_windows) == self.batch_size:
                        yielded_batches += 1
                        yield temporal_window_collate(batch_windows)
                        batch_windows = []

            if batch_windows:
                yielded_batches += 1
                yield temporal_window_collate(batch_windows)
        finally:
            self.detection_cache.flush()

        self.log_fn(
            f"Lazy {self.split_name} pass #{iteration_index + 1} produced "
            f"{materialized_windows} detector-backed temporal window(s) "
            f"across {yielded_batches} batch(es)."
        )
        if yielded_batches == 0:
            raise RuntimeError(
                f"Lazy detector produced no usable temporal windows for split '{self.split_name}'. "
                "Build more frame coverage, reduce the temporal window size, or fall back to the eager path."
            )


def _build_lazy_window_data(
    args,
    output_dir: Path,
    model_cfg,
    detector,
    camera_index: ViPECameraIndex,
    temporal_cfg: dict[str, Any],
    log_fn: Callable[[str], None],
) -> WindowDataBundle:
    frame_store = _resolve_frame_store(args)
    video_names = _resolve_lazy_video_names(args, frame_store, log_fn)

    detection_cache = args.detection_cache
    if detection_cache is None:
        video_slug = "all" if args.all_videos else "_".join(video_names)
        detection_cache = str(output_dir / f"detections_{video_slug}.json")
    log_fn(f"Using lazy detection cache path: {detection_cache}")

    frame_records = _collect_frame_records(
        frame_store,
        video_names=video_names,
        sample_limit=args.sample_limit,
    )
    if not frame_records:
        raise RuntimeError(
            "No frames were discovered for lazy fine-tuning. Build ZIP sidecar caches or provide legacy *_frames folders."
        )
    log_fn(
        f"Prepared {len(frame_records)} frame record(s) across {len(video_names)} selected video(s) for lazy detection."
    )

    train_records, val_records, split_stats = _split_frame_records(
        frame_records,
        validation_split=args.validation_split,
        seed=args.seed,
    )
    if not train_records:
        raise RuntimeError(
            "Validation split left no training frames. Reduce --validation_split or provide more frames."
        )

    train_windows, train_window_stats = _build_frame_windows(
        train_records,
        window_size=int(temporal_cfg["window_size"]),
        window_stride=int(temporal_cfg["window_stride"]),
        max_frame_gap=int(temporal_cfg["max_frame_gap"]),
    )
    if not train_windows:
        raise RuntimeError(
            "No candidate frame windows were produced for lazy fine-tuning. "
            "Increase frame coverage or reduce temporal window settings."
        )

    val_windows, val_window_stats = _build_frame_windows(
        val_records,
        window_size=int(temporal_cfg["window_size"]),
        window_stride=int(temporal_cfg["window_stride"]),
        max_frame_gap=int(temporal_cfg["max_frame_gap"]),
    )
    if not val_windows:
        val_window_stats = dict(EMPTY_WINDOW_STATS)

    detection_index = OnDemandDetectionCache(
        frame_store=frame_store,
        detector=detector,
        camera_index=camera_index,
        video_names=video_names,
        detection_conf=args.detection_conf,
        detection_cache_path=detection_cache,
        log_fn=log_fn,
    )
    crop_materializer = LazyCropMaterializer(
        model_cfg=model_cfg,
        frame_store=frame_store,
        detection_cache=detection_index,
        rescale_factor=args.rescale_factor,
        include_path_metadata=False,
        max_cached_frames=args.frame_item_cache_size,
    )

    train_iterable = LazyTemporalBatchIterable(
        split_name="train",
        frame_windows=train_windows,
        batch_size=args.batch_size,
        detection_cache=detection_index,
        crop_materializer=crop_materializer,
        shuffle=True,
        seed=args.seed,
        log_fn=log_fn,
    )
    val_iterable = None
    if val_windows:
        val_iterable = LazyTemporalBatchIterable(
            split_name="val",
            frame_windows=val_windows,
            batch_size=args.batch_size,
            detection_cache=detection_index,
            crop_materializer=crop_materializer,
            shuffle=False,
            seed=args.seed,
            log_fn=log_fn,
        )

    return WindowDataBundle(
        video_names=video_names,
        split_stats=split_stats,
        train_window_stats=train_window_stats,
        val_window_stats=val_window_stats,
        train_dataloader=train_iterable,
        val_dataloader=val_iterable,
        window_mode="lazy_frame_windows",
        sample_count_mode="lazy_detection",
    )


def _build_eager_window_data(
    args,
    output_dir: Path,
    model_cfg,
    detector,
    camera_index: ViPECameraIndex,
    temporal_cfg: dict[str, Any],
    log_fn: Callable[[str], None],
) -> WindowDataBundle:
    frame_store = _resolve_frame_store(args)
    video_names = resolve_video_names(args, log_fn)

    detection_cache = args.detection_cache
    if detection_cache is None:
        video_slug = "all" if args.all_videos else "_".join(video_names)
        detection_cache = str(output_dir / f"detections_{video_slug}.json")

    log_fn(f"Building detection samples (cache path: {detection_cache}).")
    samples = build_detection_samples(
        image_folder=args.image_folder,
        video_names=video_names,
        detector=detector,
        camera_index=camera_index,
        detection_conf=args.detection_conf,
        detection_cache_path=detection_cache,
        sample_limit=args.sample_limit,
        frame_cache_root=getattr(args, "frame_cache_root", None),
    )
    if not samples:
        raise RuntimeError("No detector samples were generated for fine-tuning.")
    log_fn(f"Collected {len(samples)} detector sample(s).")

    log_fn(f"Splitting samples into train/val with validation_split={args.validation_split}.")
    train_samples, val_samples, split_stats = split_samples_by_frame(
        samples,
        validation_split=args.validation_split,
        seed=args.seed,
    )
    if not train_samples:
        raise RuntimeError(
            "Validation split left no training samples. Reduce --validation_split or provide more frames."
        )

    log_fn(
        f"Building temporal training windows with size={temporal_cfg['window_size']}, "
        f"stride={temporal_cfg['window_stride']}, gap={temporal_cfg['max_frame_gap']}."
    )
    train_windows, train_window_stats = build_temporal_windows(
        train_samples,
        window_size=int(temporal_cfg["window_size"]),
        window_stride=int(temporal_cfg["window_stride"]),
        max_frame_gap=int(temporal_cfg["max_frame_gap"]),
    )
    if not train_windows:
        raise RuntimeError(
            "No temporal training windows were produced. Increase sample coverage or reduce temporal window settings."
        )

    log_fn("Constructing training dataset and dataloader.")
    train_base_dataset = DetectedVideoHandDataset(
        model_cfg,
        train_samples,
        rescale_factor=args.rescale_factor,
        include_path_metadata=False,
        frame_store=frame_store,
        max_cached_frames=args.frame_item_cache_size,
    )
    train_dataset = TemporalWindowDataset(train_base_dataset, train_windows)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=temporal_window_collate,
    )

    val_window_stats = dict(EMPTY_WINDOW_STATS)
    val_dataloader = None
    if val_samples:
        log_fn("Building validation temporal windows and dataloader.")
        val_windows, val_window_stats = build_temporal_windows(
            val_samples,
            window_size=int(temporal_cfg["window_size"]),
            window_stride=int(temporal_cfg["window_stride"]),
            max_frame_gap=int(temporal_cfg["max_frame_gap"]),
        )
        if val_windows:
            val_base_dataset = DetectedVideoHandDataset(
                model_cfg,
                val_samples,
                rescale_factor=args.rescale_factor,
                include_path_metadata=False,
                frame_store=frame_store,
                max_cached_frames=args.frame_item_cache_size,
            )
            val_dataset = TemporalWindowDataset(val_base_dataset, val_windows)
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False,
                collate_fn=temporal_window_collate,
            )

    return WindowDataBundle(
        video_names=video_names,
        split_stats=split_stats,
        train_window_stats=train_window_stats,
        val_window_stats=val_window_stats,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        window_mode="eager_detection",
        sample_count_mode="detected_samples",
    )


def build_window_data(
    args,
    output_dir: Path,
    model_cfg,
    detector,
    camera_index: ViPECameraIndex,
    temporal_cfg: dict[str, Any],
    log_fn: Callable[[str], None],
) -> WindowDataBundle:
    if getattr(args, "lazy_detection", True):
        log_fn(
            "Lazy detection mode is enabled. Training will assemble frame windows first and "
            "run the detector only for frames needed by each batch."
        )
        return _build_lazy_window_data(
            args=args,
            output_dir=output_dir,
            model_cfg=model_cfg,
            detector=detector,
            camera_index=camera_index,
            temporal_cfg=temporal_cfg,
            log_fn=log_fn,
        )

    log_fn(
        "Lazy detection mode is disabled. Falling back to the eager detector prepass over the selected dataset."
    )
    return _build_eager_window_data(
        args=args,
        output_dir=output_dir,
        model_cfg=model_cfg,
        detector=detector,
        camera_index=camera_index,
        temporal_cfg=temporal_cfg,
        log_fn=log_fn,
    )
