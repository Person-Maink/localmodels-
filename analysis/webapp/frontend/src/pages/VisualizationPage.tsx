import { useEffect, useMemo, useRef, useState } from "react";

import { ModeControls } from "../components/ModeControls";
import { PlaybackControls } from "../components/PlaybackControls";
import { VtkSceneCanvas } from "../components/VtkSceneCanvas";
import { api } from "../lib/api";
import type {
  FrameScene,
  ModeSpec,
  SourceAsset,
  TreeViewPayload,
  VisualizationManifest,
} from "../types";

const FRAME_WINDOW_SIZE = 12;

type VisualizationPageProps = {
  tree: TreeViewPayload;
  modes: ModeSpec[];
  selectedSources: SourceAsset[];
};

function defaultsFromMode(mode?: ModeSpec): Record<string, unknown> {
  if (!mode) {
    return {};
  }
  return Object.fromEntries(mode.controls.map((control) => [control.id, control.default ?? ""]));
}

function windowBounds(frameIds: number[], index: number) {
  const half = Math.floor(FRAME_WINDOW_SIZE / 2);
  const startIndex = Math.max(0, index - half);
  const endIndex = Math.min(frameIds.length - 1, startIndex + FRAME_WINDOW_SIZE - 1);
  return {
    startIndex,
    endIndex,
    frameStart: frameIds[startIndex],
    frameEnd: frameIds[endIndex],
  };
}

export function VisualizationPage({ tree, modes, selectedSources }: VisualizationPageProps) {
  const [selectedModeId, setSelectedModeId] = useState("");
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [manifest, setManifest] = useState<VisualizationManifest | null>(null);
  const [frames, setFrames] = useState<Record<number, FrameScene>>({});
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [selectedOverlayIds, setSelectedOverlayIds] = useState<string[]>([]);
  const [handVisibility, setHandVisibility] = useState<Record<string, boolean>>({
    left: true,
    right: true,
    unknown: true,
  });
  const [manifestError, setManifestError] = useState("");
  const [frameError, setFrameError] = useState("");
  const [loadingWindow, setLoadingWindow] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const requestedWindowsRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (!modes.length) {
      return;
    }
    if (!selectedModeId || !modes.some((mode) => mode.id === selectedModeId)) {
      setSelectedModeId(modes[0].id);
    }
  }, [modes, selectedModeId]);

  const selectedMode = useMemo(
    () => modes.find((mode) => mode.id === selectedModeId) ?? modes[0],
    [modes, selectedModeId],
  );

  useEffect(() => {
    setParams(defaultsFromMode(selectedMode));
  }, [selectedMode?.id]);

  const compatibleSources = useMemo(() => {
    if (!selectedMode?.supported_families?.length) {
      return selectedSources;
    }
    return selectedSources.filter((source) => selectedMode.supported_families.includes(source.family));
  }, [selectedMode, selectedSources]);

  const availableOverlays = useMemo(() => {
    const clipIds = new Set(compatibleSources.map((source) => source.clip_id));
    const familyIds = new Set(compatibleSources.map((source) => source.family_id));
    return Object.values(tree.overlays).filter(
      (overlay) => clipIds.has(overlay.clip_id) || familyIds.has(overlay.family_id),
    );
  }, [compatibleSources, tree.overlays]);

  useEffect(() => {
    const validOverlayIds = new Set(availableOverlays.map((overlay) => overlay.id));
    setSelectedOverlayIds((current) => current.filter((overlayId) => validOverlayIds.has(overlayId)));
  }, [availableOverlays]);

  useEffect(() => {
    if (!selectedMode || compatibleSources.length === 0) {
      setManifest(null);
      setFrames({});
      setCurrentFrameIndex(0);
      setPlaying(false);
      setManifestError("");
      return;
    }

    const overlayIds = selectedMode.id === "camera_trajectories" ? selectedOverlayIds : [];
    let cancelled = false;

    setManifestError("");
    setFrameError("");
    setLoadingWindow(false);
    setFrames({});
    setCurrentFrameIndex(0);
    requestedWindowsRef.current.clear();

    api
      .buildManifest({
        source_ids: compatibleSources.map((source) => source.id),
        overlay_ids: overlayIds,
        mode_id: selectedMode.id,
        params,
      })
      .then((nextManifest) => {
        if (cancelled) {
          return;
        }
        setManifest(nextManifest);
      })
      .catch((error) => {
        if (cancelled) {
          return;
        }
        console.error(error);
        setManifest(null);
        setFrames({});
        setManifestError(error.message || "Unable to build the visualization manifest.");
      });

    return () => {
      cancelled = true;
    };
  }, [compatibleSources, params, selectedMode, selectedOverlayIds]);

  useEffect(() => {
    if (!manifest || manifest.frame_ids.length === 0 || !selectedMode || compatibleSources.length === 0) {
      return;
    }

    const { frameStart, frameEnd } = windowBounds(manifest.frame_ids, currentFrameIndex);
    const missingAnyFrame = manifest.frame_ids
      .filter((frameId) => frameId >= frameStart && frameId <= frameEnd)
      .some((frameId) => frames[frameId] == null);

    if (!missingAnyFrame) {
      return;
    }

    const requestKey = `${manifest.request_id}:${frameStart}:${frameEnd}`;
    if (requestedWindowsRef.current.has(requestKey)) {
      return;
    }

    requestedWindowsRef.current.add(requestKey);
    setLoadingWindow(true);
    setFrameError("");

    const overlayIds = selectedMode.id === "camera_trajectories" ? selectedOverlayIds : [];
    api
      .buildFrames({
        source_ids: compatibleSources.map((source) => source.id),
        overlay_ids: overlayIds,
        mode_id: selectedMode.id,
        params,
        frame_start: frameStart,
        frame_end: frameEnd,
      })
      .then((payload) => {
        setFrames((current) => ({
          ...current,
          ...Object.fromEntries(payload.scenes.map((scene) => [scene.frame_id, scene])),
        }));
      })
      .catch((error) => {
        console.error(error);
        setFrameError(error.message || "Unable to load visualization frames.");
      })
      .finally(() => {
        requestedWindowsRef.current.delete(requestKey);
        setLoadingWindow(false);
      });
  }, [compatibleSources, currentFrameIndex, frames, manifest, params, selectedMode, selectedOverlayIds]);

  useEffect(() => {
    if (!playing || !manifest) {
      return;
    }
    const handle = window.setInterval(() => {
      setCurrentFrameIndex((current) => {
        if (!manifest.frame_ids.length) {
          return 0;
        }
        return current >= manifest.frame_ids.length - 1 ? 0 : current + 1;
      });
    }, 1000 / Math.max(1, manifest.fps || 30));
    return () => window.clearInterval(handle);
  }, [playing, manifest]);

  const currentFrameId = manifest?.frame_ids[currentFrameIndex];

  useEffect(() => {
    if (!videoRef.current || manifest == null || currentFrameId == null) {
      return;
    }
    videoRef.current.currentTime = currentFrameId / Math.max(1, manifest.fps);
  }, [currentFrameId, manifest]);

  const currentScene = currentFrameId == null ? null : frames[currentFrameId] ?? null;
  const primaryVideo = compatibleSources[0]?.video_id ? tree.videos[compatibleSources[0].video_id] : null;

  return (
    <div className="page-shell">
      <div className="page-toolbar">
        <div>
          <h2>Visualization</h2>
          <p>Compare meshes, joints, camera tracks, and averaged-beta reconstructions without loading the whole clip at once.</p>
        </div>
        <label className="mode-picker">
          <span>Mode</span>
          <select value={selectedModeId} onChange={(event) => setSelectedModeId(event.target.value)}>
            {modes.map((mode) => (
              <option key={mode.id} value={mode.id}>
                {mode.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="page-grid">
        <section className="card">
          <header className="card-header">
            <h3>Mode Controls</h3>
            <p>{selectedMode?.purpose}</p>
          </header>
          <ModeControls
            controls={selectedMode?.controls ?? []}
            values={params}
            onChange={(id, value) => setParams((current) => ({ ...current, [id]: value }))}
          />
          {selectedMode?.id === "camera_trajectories" && availableOverlays.length > 0 ? (
            <div className="overlay-panel">
              <h4>Auxiliary Overlays</h4>
              <div className="leaf-stack">
                {availableOverlays.map((overlay) => (
                  <label className="source-leaf" key={overlay.id}>
                    <input
                      type="checkbox"
                      checked={selectedOverlayIds.includes(overlay.id)}
                      onChange={() =>
                        setSelectedOverlayIds((current) =>
                          current.includes(overlay.id)
                            ? current.filter((item) => item !== overlay.id)
                            : [...current, overlay.id],
                        )
                      }
                    />
                    <div className="source-leaf-body">
                      <div className="source-leaf-head">
                        <strong>ViPE overlay</strong>
                      </div>
                      <div className="source-leaf-subtitle">{overlay.clip_id}</div>
                    </div>
                  </label>
                ))}
              </div>
            </div>
          ) : null}
          <div className="hand-toggle-row">
            {["left", "right", "unknown"].map((hand) => (
              <label key={hand}>
                <input
                  type="checkbox"
                  checked={handVisibility[hand]}
                  onChange={(event) =>
                    setHandVisibility((current) => ({ ...current, [hand]: event.target.checked }))
                  }
                />
                <span>{hand}</span>
              </label>
            ))}
          </div>
          <div className="source-color-list">
            {manifest
              ? compatibleSources.map((source) => (
                  <div className="source-color-item" key={source.id}>
                    <span
                      className="source-color-swatch"
                      style={{ backgroundColor: manifest.source_colors[source.id] ?? "#999" }}
                    />
                    <span>{source.clip_id}</span>
                  </div>
                ))
              : null}
          </div>
        </section>

        <section className="card canvas-card">
          <header className="card-header">
            <h3>3D Scene</h3>
            <p>Detected hands are always loaded. The hand checkboxes only toggle visibility.</p>
          </header>
          <VtkSceneCanvas
            staticActors={manifest?.static_actors ?? []}
            actors={currentScene?.actors ?? []}
            handVisibility={handVisibility}
          />
          <div className="status-stack">
            {compatibleSources.length === 0 ? (
              <div className="empty-state">Select at least one source from the sidebar to render a scene.</div>
            ) : null}
            {manifestError ? <div className="error-text">{manifestError}</div> : null}
            {frameError ? <div className="error-text">{frameError}</div> : null}
            {!manifestError && !frameError && loadingWindow ? (
              <div className="empty-note">Loading a small frame window around the current frame…</div>
            ) : null}
            {manifest && currentFrameId != null && currentScene == null && !loadingWindow ? (
              <div className="empty-note">No actors are loaded for frame {currentFrameId} yet.</div>
            ) : null}
          </div>
          <PlaybackControls
            playing={playing}
            canPlay={Boolean(manifest && manifest.frame_ids.length > 0)}
            currentFrameIndex={currentFrameIndex}
            maxFrameIndex={Math.max(0, (manifest?.frame_ids.length ?? 1) - 1)}
            currentFrameId={currentFrameId}
            fps={manifest?.fps}
            onPlayPause={() => setPlaying((current) => !current)}
            onStep={(delta) =>
              setCurrentFrameIndex((current) =>
                Math.min(Math.max(0, current + delta), Math.max(0, (manifest?.frame_ids.length ?? 1) - 1)),
              )
            }
            onSeek={setCurrentFrameIndex}
          />
        </section>

        <section className="card media-card">
          <header className="card-header">
            <h3>Raw Video</h3>
            <p>{primaryVideo ? primaryVideo.label : "Select a source with a matching raw video."}</p>
          </header>
          {primaryVideo ? (
            <video
              ref={videoRef}
              className="video-pane"
              src={api.videoUrl(primaryVideo.id)}
              controls
              muted
            />
          ) : (
            <div className="empty-state">No raw video resolved for the active source selection.</div>
          )}
        </section>
      </div>
    </div>
  );
}
