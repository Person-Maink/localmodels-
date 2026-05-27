import { useEffect, useMemo, useState } from "react";

import { ModeControls } from "../components/ModeControls";
import { PlotPanel } from "../components/PlotPanel";
import { SegmentEditor } from "../components/SegmentEditor";
import { api } from "../lib/api";
import type {
  AnalysisJob,
  AnalysisResult,
  ModeSpec,
  SourceAsset,
  SourceMetadata,
} from "../types";

type AnalysisPageProps = {
  modes: ModeSpec[];
  selectedSources: SourceAsset[];
};

function defaultsFromMode(mode?: ModeSpec): Record<string, unknown> {
  if (!mode) {
    return {};
  }
  return Object.fromEntries(mode.controls.map((control) => [control.id, control.default ?? ""]));
}

export function AnalysisPage({ modes, selectedSources }: AnalysisPageProps) {
  const [selectedModeId, setSelectedModeId] = useState("");
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [metadata, setMetadata] = useState<Record<string, SourceMetadata>>({});
  const [segments, setSegments] = useState<Record<string, { frame_start?: number; frame_end?: number }>>({});
  const [job, setJob] = useState<AnalysisJob | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState("");

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

  useEffect(() => {
    if (selectedSources.length === 0) {
      setMetadata({});
      return;
    }
    Promise.all(selectedSources.map((source) => api.getSourceMetadata(source.id)))
      .then((items) => {
        const next = Object.fromEntries(items.map((item) => [item.source_id, item]));
        setMetadata(next);
        setSegments((current) => {
          const merged = { ...current };
          for (const item of items) {
            if (!merged[item.source_id]) {
              merged[item.source_id] = {
                frame_start: item.first_frame_id ?? 0,
                frame_end: item.last_frame_id ?? 0,
              };
            }
          }
          return merged;
        });
      })
      .catch((nextError) => {
        console.error(nextError);
        setError(nextError.message || "Unable to load source metadata.");
      });
  }, [selectedSources]);

  useEffect(() => {
    if (!job || job.status === "succeeded" || job.status === "failed") {
      return;
    }
    const handle = window.setInterval(() => {
      api
        .getAnalysisJob(job.job_id)
        .then((nextJob) => {
          setJob(nextJob);
          if (nextJob.status === "succeeded" && nextJob.result_id) {
            api.getAnalysisResult(nextJob.result_id).then((nextResult) => {
              setResult(nextResult);
              setError("");
            });
          }
          if (nextJob.status === "failed") {
            setError(nextJob.error || "Analysis failed.");
          }
        })
        .catch((nextError) => {
          console.error(nextError);
          setError(nextError.message || "Unable to poll the analysis job.");
        });
    }, 1200);
    return () => window.clearInterval(handle);
  }, [job]);

  const compatibleSources = useMemo(() => {
    if (!selectedMode?.supported_families?.length) {
      return selectedSources;
    }
    return selectedSources.filter((source) =>
      selectedMode.supported_families.includes(source.family),
    );
  }, [selectedMode, selectedSources]);

  const selectionSummary =
    compatibleSources.length === 0
      ? "No compatible sources selected."
      : `${compatibleSources.length} source${compatibleSources.length === 1 ? "" : "s"} ready.`;

  return (
    <div className="page-shell">
      <div className="page-toolbar">
        <div>
          <h2>Frequency Analysis</h2>
          <p>Choose a segment per source, run the mode, and keep the failure state visible if a clip or hand is invalid.</p>
        </div>
        <label className="mode-picker">
          <span>Mode</span>
          <select
            value={selectedModeId}
            onChange={(event) => {
              setSelectedModeId(event.target.value);
              setResult(null);
              setJob(null);
              setError("");
            }}
          >
            {modes.map((mode) => (
              <option key={mode.id} value={mode.id}>
                {mode.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="page-grid analysis-grid">
        <section className="card">
          <header className="card-header">
            <h3>Analysis Controls</h3>
            <p>{selectedMode?.purpose}</p>
          </header>
          <ModeControls
            controls={selectedMode?.controls ?? []}
            values={params}
            onChange={(id, value) => setParams((current) => ({ ...current, [id]: value }))}
          />
          <div className="analysis-summary">{selectionSummary}</div>
          <button
            className="run-button"
            disabled={compatibleSources.length === 0}
            onClick={() => {
              setError("");
              setResult(null);
              api
                .runAnalysis({
                  mode_id: selectedMode.id,
                  params,
                  selections: compatibleSources.map((source) => ({
                    source_id: source.id,
                    frame_start: segments[source.id]?.frame_start,
                    frame_end: segments[source.id]?.frame_end,
                  })),
                })
                .then((nextJob) => {
                  setJob(nextJob);
                })
                .catch((nextError) => {
                  console.error(nextError);
                  setError(nextError.message || "Unable to start the analysis job.");
                });
            }}
          >
            Run Analysis
          </button>
          <div className="job-status">
            {job ? (
              <>
                <strong>Status:</strong> {job.status}
                {job.error ? <span className="error-text">{job.error}</span> : null}
              </>
            ) : (
              "No active analysis job."
            )}
          </div>
          {error ? <div className="error-text">{error}</div> : null}
        </section>

        <section className="card">
          <header className="card-header">
            <h3>Segment Selection</h3>
            <p>Each selected source can use a different frame window. Short ranges may not support filtering.</p>
          </header>
          <div className="segment-stack">
            {compatibleSources.map((source) => (
              <SegmentEditor
                key={source.id}
                source={source}
                metadata={metadata[source.id]}
                value={segments[source.id] ?? {}}
                onChange={(sourceId, next) =>
                  setSegments((current) => ({ ...current, [sourceId]: next }))
                }
              />
            ))}
            {compatibleSources.length === 0 ? (
              <div className="empty-state">Select at least one compatible source in the sidebar.</div>
            ) : null}
          </div>
        </section>

        <section className="card results-card">
          <header className="card-header">
            <h3>Results</h3>
            <p>Transient SVG exports are available after a successful run.</p>
          </header>
          {result ? (
            <>
              <PlotPanel result={result} />
              <div className="metrics-table">
                <table>
                  <thead>
                    <tr>
                      <th>Label</th>
                      <th>Hand Used</th>
                      <th>Dominant Hz</th>
                      <th>Peak Ratio</th>
                      <th>Peak Sharpness</th>
                      <th>Temporal Noise</th>
                      <th>Spatial Coherence</th>
                      <th>RMS</th>
                      <th>Samples</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.entries.map((entry) => (
                      <tr key={`${entry.label}:${entry.pair_label ?? "default"}`}>
                        <td>{entry.label}</td>
                        <td>{entry.hand_used ?? "-"}</td>
                        <td>{entry.dominant_hz?.toFixed(3) ?? "-"}</td>
                        <td>{entry.peak_ratio?.toExponential(3) ?? "-"}</td>
                        <td>{entry.peak_sharpness?.toExponential(3) ?? "-"}</td>
                        <td>{entry.temporal_noise?.toExponential(3) ?? "-"}</td>
                        <td>{entry.spatial_coherence == null ? "-" : entry.spatial_coherence.toExponential(3)}</td>
                        <td>{entry.rms_amplitude?.toExponential(3) ?? "-"}</td>
                        <td>{entry.sample_count ?? "-"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {result.figure_svg_url ? (
                <a className="download-link" href={result.figure_svg_url} target="_blank" rel="noreferrer">
                  Open SVG export
                </a>
              ) : null}
            </>
          ) : (
            <div className="empty-state">Run an analysis to populate plots, metrics, and the export link.</div>
          )}
        </section>
      </div>
    </div>
  );
}
