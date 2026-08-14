from __future__ import annotations

import base64
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse

from .settings import AppSettings, load_settings

settings = load_settings()
os.environ.setdefault("MPLCONFIGDIR", str(settings.temp_root / "mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from .analysis import AnalysisService
from .cache import LruCache
from .catalog import LibraryCatalog, build_library_catalog
from .dynhamr import normalize_dynhamr_outputs
from .loaders import source_metadata
from .models import (
    AnalysisJobResponse,
    AnalysisJobStatus,
    AnalysisResultResponse,
    AnalysisRunRequest,
    VisualizationFrameRequest,
    VisualizationManifestRequest,
)
from .registry import analysis_modes, visualization_modes
from .visualization import VisualizationService


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AppState:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.catalog: LibraryCatalog = build_library_catalog(settings, scanned_at=_utc_now())
        self.visualization = VisualizationService(settings)
        self.analysis = AnalysisService(settings)
        self.analysis_executor = ThreadPoolExecutor(max_workers=2)
        self.analysis_jobs: Dict[str, dict] = {}
        self.analysis_results: Dict[str, dict] = {}
        self.analysis_svgs: LruCache[str, str] = LruCache(max_items=settings.max_cache_items)

    def rescan(self, overwrite_dynhamr: bool = False) -> dict:
        summaries = normalize_dynhamr_outputs(self.settings, overwrite=overwrite_dynhamr)
        self.catalog = build_library_catalog(self.settings, scanned_at=_utc_now())
        self.visualization.manifest_cache.clear()
        self.visualization.frame_cache.clear()
        return {"normalized_dynhamr_runs": summaries, "scanned_at": self.catalog.scanned_at}

    def require_sources(self, source_ids: List[str]) -> List[dict]:
        sources = []
        for source_id in source_ids:
            source = self.catalog.sources.get(source_id)
            if source is None:
                raise HTTPException(status_code=404, detail=f"Unknown source_id: {source_id}")
            sources.append(source)
        return sources

    def resolve_overlays(self, overlay_ids: List[str]) -> List[dict]:
        overlays = []
        for overlay_id in overlay_ids:
            overlay = self.catalog.overlays.get(overlay_id)
            if overlay is None:
                raise HTTPException(status_code=404, detail=f"Unknown overlay_id: {overlay_id}")
            overlays.append(overlay)
        return overlays


state = AppState(settings=settings)

app = FastAPI(title="Tremor Analysis Web App", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"ok": True, "scanned_at": state.catalog.scanned_at}


@app.get("/api/library/tree")
def get_library_tree():
    return state.catalog.to_response()


@app.post("/api/library/rescan")
def rescan_library(overwrite_dynhamr: bool = False):
    return state.rescan(overwrite_dynhamr=overwrite_dynhamr)


@app.get("/api/modes/visualization")
def get_visualization_modes():
    return [mode.model_dump() for mode in visualization_modes()]


@app.get("/api/modes/analysis")
def get_analysis_modes():
    return [mode.model_dump() for mode in analysis_modes()]


@app.get("/api/assets/video/{video_id}")
def get_video_asset(video_id: str):
    video = state.catalog.videos.get(video_id)
    if video is None:
        raise HTTPException(status_code=404, detail=f"Unknown video_id: {video_id}")
    return FileResponse(video["path"], media_type="video/mp4", filename=f"{video['stem']}.mp4")


@app.get("/api/sources/{source_id}/metadata")
def get_source_metadata(source_id: str):
    source = state.catalog.sources.get(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail=f"Unknown source_id: {source_id}")
    return source_metadata(source, settings=state.settings)


@app.post("/api/visualization/manifest")
def build_visualization_manifest(request: VisualizationManifestRequest):
    sources = state.require_sources(request.source_ids)
    overlays = state.resolve_overlays(request.overlay_ids)
    return state.visualization.build_manifest(request.mode_id, sources, overlays, request.params).model_dump()


@app.post("/api/visualization/frames")
def build_visualization_frames(request: VisualizationFrameRequest):
    sources = state.require_sources(request.source_ids)
    overlays = state.resolve_overlays(request.overlay_ids)
    return state.visualization.build_frames(
        mode_id=request.mode_id,
        sources=sources,
        overlays=overlays,
        params=request.params,
        frame_start=request.frame_start,
        frame_end=request.frame_end,
    ).model_dump()


def _submit_analysis_job(job_id: str, mode_id: str, sources: List[dict], frame_ranges: Dict[str, tuple], params: dict) -> None:
    try:
        payload, svg = state.analysis.run(mode_id, sources, frame_ranges, params)
        result_id = uuid.uuid4().hex
        svg_key = f"{result_id}.svg"
        state.analysis_svgs.set(svg_key, svg)
        state.analysis_results[result_id] = {
            "result_id": result_id,
            "mode_id": mode_id,
            "title": payload["title"],
            "inputs": {
                "source_ids": [source["id"] for source in sources],
                "frame_ranges": frame_ranges,
                "params": params,
            },
            "entries": payload.get("entries", []),
            "tables": payload.get("tables", []),
            "metrics": payload.get("metrics", {}),
            "figure_svg_url": f"/api/analysis/results/{result_id}/figure.svg",
        }
        state.analysis_jobs[job_id] = {"job_id": job_id, "status": "succeeded", "result_id": result_id, "error": None}
    except Exception as exc:  # noqa: BLE001
        state.analysis_jobs[job_id] = {"job_id": job_id, "status": "failed", "result_id": None, "error": str(exc)}


@app.post("/api/analysis/run", response_model=AnalysisJobResponse)
def run_analysis(request: AnalysisRunRequest):
    sources = state.require_sources([selection.source_id for selection in request.selections])
    source_map = {source["id"]: source for source in sources}
    frame_ranges = {
        selection.source_id: (selection.frame_start, selection.frame_end)
        for selection in request.selections
    }

    mode_lookup = {mode.id: mode for mode in analysis_modes()}
    mode = mode_lookup.get(request.mode_id)
    if mode is None:
        raise HTTPException(status_code=404, detail=f"Unknown analysis mode: {request.mode_id}")
    if len(sources) < mode.min_sources:
        raise HTTPException(status_code=400, detail=f"{mode.label} requires at least {mode.min_sources} source(s).")
    if mode.max_sources is not None and len(sources) > mode.max_sources:
        raise HTTPException(status_code=400, detail=f"{mode.label} supports at most {mode.max_sources} source(s).")
    for source in source_map.values():
        if mode.supported_families and source["family"] not in mode.supported_families:
            raise HTTPException(status_code=400, detail=f"{mode.label} does not support family '{source['family']}'.")

    job_id = uuid.uuid4().hex
    state.analysis_jobs[job_id] = {"job_id": job_id, "status": "queued", "result_id": None, "error": None}

    def runner():
        state.analysis_jobs[job_id]["status"] = "running"
        _submit_analysis_job(job_id, request.mode_id, sources, frame_ranges, request.params)

    state.analysis_executor.submit(runner)
    return AnalysisJobResponse(job_id=job_id, status="queued")


@app.get("/api/analysis/jobs/{job_id}", response_model=AnalysisJobStatus)
def get_analysis_job(job_id: str):
    payload = state.analysis_jobs.get(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
    return AnalysisJobStatus(**payload)


@app.get("/api/analysis/results/{result_id}", response_model=AnalysisResultResponse)
def get_analysis_result(result_id: str):
    payload = state.analysis_results.get(result_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Unknown result_id: {result_id}")
    return AnalysisResultResponse(**payload)


@app.get("/api/analysis/results/{result_id}/figure.svg")
def get_analysis_svg(result_id: str):
    payload = state.analysis_results.get(result_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Unknown result_id: {result_id}")
    svg_key = f"{result_id}.svg"
    svg = state.analysis_svgs.get(svg_key)
    if svg is None:
        raise HTTPException(status_code=404, detail="SVG payload expired from transient cache.")
    return PlainTextResponse(svg, media_type="image/svg+xml")


@app.get("/api/debug/settings")
def debug_settings():
    return {
        "repo_root": str(state.settings.repo_root),
        "workspace_root": str(state.settings.workspace_root),
        "data_root": str(state.settings.data_root),
        "outputs_root": str(state.settings.outputs_root),
        "dynhamr_logs_root": str(state.settings.dynhamr_logs_root),
        "dynhamr_output_root": str(state.settings.dynhamr_output_root),
        "mediapipe_root": str(state.settings.mediapipe_root),
        "vipe_pose_root": str(state.settings.vipe_pose_root),
        "mano_right_path": str(state.settings.mano_right_path),
    }


def run_dev():
    import uvicorn

    uvicorn.run("webapp.backend.main:app", host="127.0.0.1", port=8000, reload=True)
