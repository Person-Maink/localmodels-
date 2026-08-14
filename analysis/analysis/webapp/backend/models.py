from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CapabilityFlags(BaseModel):
    visualization: bool = True
    frequency: bool = True
    camera: bool = False
    bounding_boxes: bool = False
    beta: bool = False
    auxiliary_only: bool = False


class VideoAsset(BaseModel):
    id: str
    label: str
    stem: str
    path: str


class SourceAsset(BaseModel):
    id: str
    family: str
    label: str
    clip_id: str
    family_id: str
    path: str
    experiment: Optional[str] = None
    video_id: Optional[str] = None
    capabilities: CapabilityFlags


class OverlayAsset(BaseModel):
    id: str
    kind: str
    clip_id: str
    family_id: str
    pose_path: str
    rgb_video_id: Optional[str] = None


class ControlSpec(BaseModel):
    id: str
    label: str
    type: str
    required: bool = False
    default: Optional[Any] = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: List[Dict[str, Any]] = Field(default_factory=list)
    placeholder: Optional[str] = None
    help: Optional[str] = None


class ModeSpec(BaseModel):
    id: str
    label: str
    tab: str
    script_name: str
    purpose: str
    required_inputs: List[str]
    optional_arguments: List[str]
    expected_outputs: List[str]
    ui_appearance: str
    min_sources: int = 1
    max_sources: Optional[int] = None
    supported_families: List[str] = Field(default_factory=list)
    controls: List[ControlSpec] = Field(default_factory=list)


class VisualizationManifestRequest(BaseModel):
    source_ids: List[str]
    overlay_ids: List[str] = Field(default_factory=list)
    mode_id: str
    params: Dict[str, Any] = Field(default_factory=dict)


class VisualizationFrameRequest(BaseModel):
    source_ids: List[str]
    overlay_ids: List[str] = Field(default_factory=list)
    mode_id: str
    params: Dict[str, Any] = Field(default_factory=dict)
    frame_start: int = 0
    frame_end: int = 0


class SceneActor(BaseModel):
    id: str
    kind: str
    label: str
    source_id: Optional[str] = None
    hand: Optional[str] = None
    color: str = "#4c6ef5"
    opacity: float = 1.0
    points: List[List[float]] = Field(default_factory=list)
    faces: List[List[int]] = Field(default_factory=list)
    segments: List[List[List[float]]] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class FrameScene(BaseModel):
    frame_id: int
    actors: List[SceneActor] = Field(default_factory=list)


class VisualizationManifestResponse(BaseModel):
    request_id: str
    mode_id: str
    fps: float
    frame_ids: List[int]
    source_colors: Dict[str, str]
    available_hands: List[str]
    static_actors: List[SceneActor] = Field(default_factory=list)
    camera_display: Dict[str, Any] = Field(default_factory=dict)


class VisualizationFrameResponse(BaseModel):
    request_id: str
    mode_id: str
    scenes: List[FrameScene]


class SourceSelection(BaseModel):
    source_id: str
    frame_start: Optional[int] = None
    frame_end: Optional[int] = None


class AnalysisRunRequest(BaseModel):
    mode_id: str
    selections: List[SourceSelection]
    params: Dict[str, Any] = Field(default_factory=dict)


class AnalysisJobResponse(BaseModel):
    job_id: str
    status: str


class AnalysisJobStatus(BaseModel):
    job_id: str
    status: str
    result_id: Optional[str] = None
    error: Optional[str] = None


class AnalysisResultResponse(BaseModel):
    result_id: str
    mode_id: str
    title: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    entries: List[Dict[str, Any]] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    figure_svg_url: Optional[str] = None
