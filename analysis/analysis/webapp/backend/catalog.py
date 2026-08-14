from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .settings import (
    AppSettings,
    BETA_COMPATIBLE_FAMILIES,
    CAMERA_COMPATIBLE_FAMILIES,
    MODEL_FAMILIES,
    PRIMARY_FAMILIES,
    VIDEO_EXTENSIONS,
    strip_variant_suffix,
)


@dataclass
class LibraryCatalog:
    scanned_at: str
    sources: Dict[str, dict]
    videos: Dict[str, dict]
    overlays: Dict[str, dict]
    by_clip: List[dict]
    by_source: List[dict]
    by_experiment: List[dict]

    def to_response(self) -> dict:
        return {
            "scanned_at": self.scanned_at,
            "organization_modes": ["by_clip", "by_source", "by_experiment"],
            "sources": self.sources,
            "videos": self.videos,
            "overlays": self.overlays,
            "views": {
                "by_clip": self.by_clip,
                "by_source": self.by_source,
                "by_experiment": self.by_experiment,
            },
        }


def _source_capabilities(family: str) -> dict:
    capabilities = {
        "visualization": True,
        "frequency": family in PRIMARY_FAMILIES,
        "camera": family in CAMERA_COMPATIBLE_FAMILIES,
        "bounding_boxes": family in MODEL_FAMILIES,
        "beta": family in BETA_COMPATIBLE_FAMILIES,
        "auxiliary_only": False,
    }
    if family == "mediapipe":
        capabilities["camera"] = False
        capabilities["bounding_boxes"] = False
        capabilities["beta"] = False
    return capabilities


def _video_asset_id(stem: str) -> str:
    return f"video:{stem}"


def _source_id(family: str, clip_id: str, experiment: Optional[str] = None) -> str:
    if experiment:
        return f"{family}:{experiment}:{clip_id}"
    return f"{family}:{clip_id}"


def _overlay_id(kind: str, clip_id: str) -> str:
    return f"{kind}:{clip_id}"


def _iter_video_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return ()
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def _scan_videos(settings: AppSettings) -> Dict[str, dict]:
    video_files = {}
    fallback_roots = [
        settings.data_root,
        settings.vipe_rgb_root,
        settings.outputs_root / "hamba" / "videos",
        settings.outputs_root / "wilor" / "videos",
    ]
    for root in fallback_roots:
        for path in _iter_video_files(root):
            stem = path.stem
            asset_id = _video_asset_id(stem)
            if asset_id in video_files:
                continue
            video_files[asset_id] = {
                "id": asset_id,
                "label": stem,
                "stem": stem,
                "path": str(path.resolve()),
            }
    return dict(sorted(video_files.items()))


def _scan_wilor(root: Path, family: str) -> Iterable[Tuple[str, Path, Optional[str]]]:
    if not root.is_dir():
        return ()
    return (
        (clip_dir.name, clip_dir / "meshes", None)
        for clip_dir in sorted(root.iterdir())
        if clip_dir.is_dir() and (clip_dir / "meshes").is_dir()
    )


def _scan_wilor_finetune(root: Path) -> Iterable[Tuple[str, Path, Optional[str]]]:
    if not root.is_dir():
        return ()
    rows = []
    for experiment_dir in sorted(root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        for clip_dir in sorted(experiment_dir.iterdir()):
            if clip_dir.is_dir() and (clip_dir / "meshes").is_dir():
                rows.append((clip_dir.name, clip_dir / "meshes", experiment_dir.name))
    return rows


def _scan_stride(root: Path) -> Iterable[Tuple[str, Path, Optional[str]]]:
    if not root.is_dir():
        return ()
    return (
        (clip_dir.name, clip_dir, None)
        for clip_dir in sorted(root.iterdir())
        if clip_dir.is_dir() and (clip_dir / "refined_sequence.npz").is_file()
    )


def _scan_mediapipe(root: Path) -> Iterable[Tuple[str, Path, Optional[str]]]:
    if not root.is_dir():
        return ()
    csv_paths = list(root.glob("*.csv")) + list((root / "keypoints").glob("*.csv"))
    rows = []
    for csv_path in sorted(csv_paths):
        clip_id = csv_path.stem
        if clip_id.endswith("_keypoints"):
            clip_id = clip_id[: -len("_keypoints")]
        rows.append((clip_id, csv_path, None))
    return rows


def _scan_family_sources(settings: AppSettings) -> Dict[str, dict]:
    rows = []
    family_roots = {
        "wilor": settings.outputs_root / "wilor",
        "hamba": settings.outputs_root / "hamba",
        "dynhamr": settings.dynhamr_output_root,
    }
    for family, root in family_roots.items():
        rows.extend((family, clip_id, path, experiment) for clip_id, path, experiment in _scan_wilor(root, family))

    rows.extend(
        ("wilor_finetune", clip_id, path, experiment)
        for clip_id, path, experiment in _scan_wilor_finetune(settings.outputs_root / "wilor_finetune")
    )
    rows.extend(("stride", clip_id, path, experiment) for clip_id, path, experiment in _scan_stride(settings.outputs_root / "stride"))
    rows.extend(
        ("mediapipe", clip_id, path, experiment)
        for clip_id, path, experiment in _scan_mediapipe(settings.mediapipe_root)
    )

    videos = _scan_videos(settings)
    by_stem = {item["stem"]: item["id"] for item in videos.values()}
    sources = {}
    for family, clip_id, path, experiment in rows:
        family_id = strip_variant_suffix(clip_id)
        source_id = _source_id(family, clip_id, experiment=experiment)
        video_id = by_stem.get(clip_id) or by_stem.get(family_id)
        label = f"{family} ({experiment})" if experiment else family
        sources[source_id] = {
            "id": source_id,
            "family": family,
            "label": label,
            "clip_id": clip_id,
            "family_id": family_id,
            "path": str(Path(path).resolve()),
            "experiment": experiment,
            "video_id": video_id,
            "capabilities": _source_capabilities(family),
        }
    return dict(sorted(sources.items()))


def _scan_vipe_overlays(settings: AppSettings, videos: Dict[str, dict]) -> Dict[str, dict]:
    overlays = {}
    by_stem = {item["stem"]: item["id"] for item in videos.values()}
    if not settings.vipe_pose_root.is_dir():
        return overlays
    for path in sorted(settings.vipe_pose_root.glob("*.npz")):
        clip_id = path.stem
        overlay_id = _overlay_id("vipe", clip_id)
        overlays[overlay_id] = {
            "id": overlay_id,
            "kind": "vipe",
            "clip_id": clip_id,
            "family_id": strip_variant_suffix(clip_id),
            "pose_path": str(path.resolve()),
            "rgb_video_id": by_stem.get(clip_id),
        }
    return overlays


def _group_by_clip(sources: Dict[str, dict], overlays: Dict[str, dict]) -> List[dict]:
    families: Dict[str, dict] = {}
    for source in sources.values():
        family_bucket = families.setdefault(
            source["family_id"],
            {
                "id": source["family_id"],
                "label": source["family_id"],
                "variants": {},
                "overlay_ids": [],
            },
        )
        variant = family_bucket["variants"].setdefault(
            source["clip_id"],
            {
                "id": source["clip_id"],
                "label": source["clip_id"],
                "video_id": source["video_id"],
                "source_ids": [],
                "experiments": [],
            },
        )
        variant["source_ids"].append(source["id"])
        if source["experiment"] and source["experiment"] not in variant["experiments"]:
            variant["experiments"].append(source["experiment"])

    for overlay in overlays.values():
        family_bucket = families.setdefault(
            overlay["family_id"],
            {"id": overlay["family_id"], "label": overlay["family_id"], "variants": {}, "overlay_ids": []},
        )
        family_bucket["overlay_ids"].append(overlay["id"])

    rows = []
    for family_id in sorted(families):
        payload = families[family_id]
        payload["variants"] = [
            payload["variants"][clip_id]
            for clip_id in sorted(payload["variants"])
        ]
        rows.append(payload)
    return rows


def _group_by_source(sources: Dict[str, dict]) -> List[dict]:
    groups: Dict[str, dict] = {}
    for source in sources.values():
        group = groups.setdefault(
            source["family"],
            {"id": source["family"], "label": source["family"], "experiments": {}},
        )
        experiment_key = source["experiment"] or "default"
        experiment = group["experiments"].setdefault(
            experiment_key,
            {"id": experiment_key, "label": experiment_key, "source_ids": []},
        )
        experiment["source_ids"].append(source["id"])

    rows = []
    for family in sorted(groups):
        payload = groups[family]
        payload["experiments"] = [
            payload["experiments"][experiment]
            for experiment in sorted(payload["experiments"])
        ]
        rows.append(payload)
    return rows


def _group_by_experiment(sources: Dict[str, dict]) -> List[dict]:
    groups: Dict[str, dict] = {}
    for source in sources.values():
        experiment_key = source["experiment"] or "default"
        group = groups.setdefault(
            experiment_key,
            {"id": experiment_key, "label": experiment_key, "source_ids": []},
        )
        group["source_ids"].append(source["id"])
    return [groups[key] for key in sorted(groups)]


def build_library_catalog(settings: AppSettings, scanned_at: str) -> LibraryCatalog:
    videos = _scan_videos(settings)
    sources = _scan_family_sources(settings)
    overlays = _scan_vipe_overlays(settings, videos)
    return LibraryCatalog(
        scanned_at=scanned_at,
        sources=sources,
        videos=videos,
        overlays=overlays,
        by_clip=_group_by_clip(sources, overlays),
        by_source=_group_by_source(sources),
        by_experiment=_group_by_experiment(sources),
    )
