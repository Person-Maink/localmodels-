import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_VIDEO_EXTS = (".mp4", ".avi", ".mts", ".mov")
SUPPORTED_IMAGE_EXTS = (".jpg", ".jpeg", ".png")
DEFAULT_JPEG_QUALITY = 95


@dataclass(frozen=True)
class FrameRecord:
    video_name: str
    frame_id: int
    frame_name: str
    source_key: str
    source_kind: str


@dataclass
class _VideoSource:
    video_name: str
    source_kind: str
    frame_records: list[FrameRecord]
    raw_video_path: Path | None = None
    zip_path: Path | None = None
    index_path: Path | None = None
    loose_dir: Path | None = None

    def get_source_path(self) -> Path | None:
        if self.raw_video_path is not None:
            return self.raw_video_path
        if self.loose_dir is not None:
            return self.loose_dir
        if self.zip_path is not None:
            return self.zip_path
        if self.frame_records:
            return Path(self.frame_records[0].source_key)
        return None


def _sorted_image_paths(root: Path) -> list[Path]:
    return sorted(
        path for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS
    )


def _decode_image_bytes(buffer: bytes) -> np.ndarray | None:
    arr = np.frombuffer(buffer, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _parse_frame_id(path: Path, fallback: int) -> int:
    stem = path.stem
    if stem.startswith("frame_"):
        try:
            return int(stem.split("_")[-1])
        except ValueError:
            return fallback
    return fallback


def _load_cache_index(zip_path: Path, index_path: Path) -> list[FrameRecord]:
    with open(index_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"Invalid frame index payload: {index_path}")

    with zipfile.ZipFile(zip_path, "r") as archive:
        names = set(archive.namelist())

    records: list[FrameRecord] = []
    expected_ids: list[int] = []
    for entry in frames:
        frame_id = int(entry["frame_id"])
        frame_name = str(entry["frame_name"])
        member_name = str(entry["member_name"])
        if member_name not in names:
            raise ValueError(f"Indexed ZIP member is missing: {member_name}")
        records.append(
            FrameRecord(
                video_name=str(payload["video_name"]),
                frame_id=frame_id,
                frame_name=frame_name,
                source_key=member_name,
                source_kind="zip",
            )
        )
        expected_ids.append(frame_id)

    sorted_ids = sorted(expected_ids)
    if sorted_ids != expected_ids:
        raise ValueError(f"Frame index is not ordered for {index_path}")
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError(f"Duplicate frame IDs found in {index_path}")
    return records


def validate_video_frame_cache(zip_path: str | Path, index_path: str | Path) -> list[FrameRecord]:
    zip_path = Path(zip_path)
    index_path = Path(index_path)
    if not zip_path.is_file():
        raise FileNotFoundError(f"Missing frame ZIP cache: {zip_path}")
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing frame index file: {index_path}")
    return _load_cache_index(zip_path, index_path)


def build_video_frame_cache(
    video_path: str | Path,
    cache_root: str | Path | None = None,
    overwrite: bool = False,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> dict:
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Missing source video: {video_path}")

    cache_root = Path(cache_root) if cache_root is not None else video_path.parent
    cache_root.mkdir(parents=True, exist_ok=True)
    video_name = video_path.stem
    zip_path = cache_root / f"{video_name}.frames.zip"
    index_path = cache_root / f"{video_name}.frames.index.json"

    if not overwrite:
        try:
            records = validate_video_frame_cache(zip_path, index_path)
            return {
                "video_name": video_name,
                "status": "skipped",
                "frame_count": len(records),
                "zip_path": str(zip_path),
                "index_path": str(index_path),
            }
        except (FileNotFoundError, ValueError):
            pass

    tmp_zip_path = zip_path.with_suffix(zip_path.suffix + ".tmp")
    tmp_index_path = index_path.with_suffix(index_path.suffix + ".tmp")
    tmp_zip_path.unlink(missing_ok=True)
    tmp_index_path.unlink(missing_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video for cache building: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps_value = fps if np.isfinite(fps) and fps > 0 else None
    frames_payload: list[dict] = []

    try:
        with zipfile.ZipFile(tmp_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            frame_id = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_name = f"frame_{frame_id:06d}"
                member_name = f"{frame_name}.jpg"
                encoded_ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                )
                if not encoded_ok:
                    raise RuntimeError(f"Failed to JPEG-encode frame {frame_id} from {video_path}")
                archive.writestr(member_name, encoded.tobytes())
                frames_payload.append(
                    {
                        "frame_id": frame_id,
                        "frame_name": frame_name,
                        "member_name": member_name,
                        "width": int(frame.shape[1]),
                        "height": int(frame.shape[0]),
                    }
                )
                frame_id += 1
    except Exception:
        tmp_zip_path.unlink(missing_ok=True)
        tmp_index_path.unlink(missing_ok=True)
        raise
    finally:
        capture.release()

    if not frames_payload:
        tmp_zip_path.unlink(missing_ok=True)
        raise RuntimeError(f"Video produced no readable frames: {video_path}")

    with zipfile.ZipFile(tmp_zip_path, "r") as archive:
        names = sorted(archive.namelist())
    expected_names = [entry["member_name"] for entry in frames_payload]
    if names != expected_names:
        tmp_zip_path.unlink(missing_ok=True)
        raise RuntimeError(f"Built ZIP members do not match expected frame set for {video_path}")

    payload = {
        "video_name": video_name,
        "source_video": video_path.name,
        "frame_count": len(frames_payload),
        "fps": fps_value,
        "frames": frames_payload,
    }
    with open(tmp_index_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    zip_path.unlink(missing_ok=True)
    index_path.unlink(missing_ok=True)
    tmp_zip_path.replace(zip_path)
    tmp_index_path.replace(index_path)

    return {
        "video_name": video_name,
        "status": "built",
        "frame_count": len(frames_payload),
        "zip_path": str(zip_path),
        "index_path": str(index_path),
    }


def build_frame_caches(
    input_root: str | Path,
    video_name: str | None = None,
    overwrite: bool = False,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> list[dict]:
    input_root = Path(input_root)
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {input_root}")

    video_paths = sorted(
        path for path in input_root.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTS
    )
    if video_name is not None:
        video_paths = [path for path in video_paths if path.stem == video_name]
        if not video_paths:
            raise FileNotFoundError(f"Could not find video '{video_name}' under {input_root}")

    return [
        build_video_frame_cache(
            video_path,
            cache_root=input_root,
            overwrite=overwrite,
            jpeg_quality=jpeg_quality,
        )
        for video_path in video_paths
    ]


class FrameStore:
    def __init__(self, image_root: str | Path, cache_root: str | Path | None = None):
        self.image_root = Path(image_root)
        self.cache_root = Path(cache_root) if cache_root is not None else self.image_root
        self._sources: dict[str, _VideoSource] = {}
        self._unavailable_videos: dict[str, str] = {}
        self._scan()

    def _scan(self) -> None:
        if not self.image_root.is_dir():
            raise NotADirectoryError(f"Image root is not a directory: {self.image_root}")

        raw_videos = {
            path.stem: path
            for path in sorted(self.image_root.iterdir())
            if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTS
        }
        loose_dirs = {
            path.name[: -len("_frames")]: path
            for path in sorted(self.image_root.glob("*_frames"))
            if path.is_dir()
        }

        for video_name, video_path in raw_videos.items():
            zip_path = self.cache_root / f"{video_name}.frames.zip"
            index_path = self.cache_root / f"{video_name}.frames.index.json"
            if zip_path.exists() or index_path.exists():
                try:
                    records = validate_video_frame_cache(zip_path, index_path)
                    self._sources[video_name] = _VideoSource(
                        video_name=video_name,
                        source_kind="zip",
                        frame_records=records,
                        raw_video_path=video_path,
                        zip_path=zip_path,
                        index_path=index_path,
                    )
                    continue
                except (FileNotFoundError, ValueError) as exc:
                    if video_name not in loose_dirs:
                        self._unavailable_videos[video_name] = (
                            f"Invalid sidecar cache for '{video_name}': {exc}"
                        )
                        continue

            if video_name in loose_dirs:
                frame_paths = sorted(
                    path for path in loose_dirs[video_name].iterdir()
                    if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS
                )
                records = [
                    FrameRecord(
                        video_name=video_name,
                        frame_id=_parse_frame_id(path, index),
                        frame_name=path.stem,
                        source_key=str(path),
                        source_kind="loose_frames",
                    )
                    for index, path in enumerate(frame_paths)
                ]
                records.sort(key=lambda record: int(record.frame_id))
                self._sources[video_name] = _VideoSource(
                    video_name=video_name,
                    source_kind="loose_frames",
                    frame_records=records,
                    raw_video_path=video_path,
                    loose_dir=loose_dirs[video_name],
                )
            else:
                self._unavailable_videos[video_name] = (
                    f"Found raw video '{video_path.name}' but no valid sidecar cache or legacy "
                    f"'{video_name}_frames' directory. Build the cache first."
                )

        for video_name, frame_dir in loose_dirs.items():
            if video_name in self._sources:
                continue
            frame_paths = sorted(
                path for path in frame_dir.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS
            )
            records = [
                FrameRecord(
                    video_name=video_name,
                    frame_id=_parse_frame_id(path, index),
                    frame_name=path.stem,
                    source_key=str(path),
                    source_kind="loose_frames",
                )
                for index, path in enumerate(frame_paths)
            ]
            records.sort(key=lambda record: int(record.frame_id))
            self._sources[video_name] = _VideoSource(
                video_name=video_name,
                source_kind="loose_frames",
                frame_records=records,
                loose_dir=frame_dir,
            )

        single_images = _sorted_image_paths(self.image_root)
        if single_images:
            self._sources["single_images"] = _VideoSource(
                video_name="single_images",
                source_kind="single_image",
                frame_records=[
                    FrameRecord(
                        video_name="single_images",
                        frame_id=index,
                        frame_name=path.stem,
                        source_key=str(path),
                        source_kind="single_image",
                    )
                    for index, path in enumerate(single_images)
                ],
            )

    def list_videos(self) -> list[str]:
        return sorted(self._sources)

    def has_video(self, video_name: str) -> bool:
        return video_name in self._sources

    def explain_unavailable(self, video_name: str) -> str | None:
        return self._unavailable_videos.get(video_name)

    def iter_video_frames(self, video_name: str):
        source = self._sources.get(video_name)
        if source is None:
            raise KeyError(f"Unknown video source: {video_name}")
        yield from source.frame_records

    def get_frame_name(self, video_name: str, frame_id: int) -> str:
        source = self._sources.get(video_name)
        if source is None:
            raise KeyError(f"Unknown video source: {video_name}")
        for record in source.frame_records:
            if int(record.frame_id) == int(frame_id):
                return record.frame_name
        raise KeyError(f"Unknown frame {frame_id} for video {video_name}")

    def get_source_path(self, video_name: str) -> Path | None:
        source = self._sources.get(video_name)
        if source is None:
            return None
        return source.get_source_path()

    def get_frame(self, video_name: str, frame_id: int) -> np.ndarray | None:
        source = self._sources.get(video_name)
        if source is None:
            raise KeyError(f"Unknown video source: {video_name}")
        record = next(
            (frame for frame in source.frame_records if int(frame.frame_id) == int(frame_id)),
            None,
        )
        if record is None:
            raise KeyError(f"Unknown frame {frame_id} for video {video_name}")

        if source.source_kind == "zip":
            assert source.zip_path is not None
            with zipfile.ZipFile(source.zip_path, "r") as archive:
                return _decode_image_bytes(archive.read(record.source_key))
        if source.source_kind in {"loose_frames", "single_image"}:
            return cv2.imread(record.source_key)
        raise ValueError(f"Unsupported source kind: {source.source_kind}")
