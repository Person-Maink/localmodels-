import type { SourceAsset, SourceMetadata } from "../types";

type SegmentEditorProps = {
  source: SourceAsset;
  metadata?: SourceMetadata;
  value: { frame_start?: number; frame_end?: number };
  onChange: (sourceId: string, next: { frame_start?: number; frame_end?: number }) => void;
};

export function SegmentEditor({ source, metadata, value, onChange }: SegmentEditorProps) {
  const firstFrame = metadata?.first_frame_id ?? 0;
  const lastFrame = metadata?.last_frame_id ?? Math.max(firstFrame, (metadata?.frame_count ?? 1) - 1);
  const frameStart = value.frame_start ?? firstFrame;
  const frameEnd = value.frame_end ?? lastFrame;

  return (
    <div className="segment-card">
      <div>
        <h4>{source.clip_id}</h4>
        <p>
          {source.family}
          {source.experiment ? ` / ${source.experiment}` : ""}
        </p>
      </div>
      <div className="segment-grid">
        <label>
          <span>Start frame</span>
          <input
            type="number"
            min={firstFrame}
            max={lastFrame}
            value={frameStart}
            onChange={(event) =>
              onChange(source.id, {
                frame_start: Number(event.target.value),
                frame_end: frameEnd
              })
            }
          />
        </label>
        <label>
          <span>End frame</span>
          <input
            type="number"
            min={firstFrame}
            max={lastFrame}
            value={frameEnd}
            onChange={(event) =>
              onChange(source.id, {
                frame_start: frameStart,
                frame_end: Number(event.target.value)
              })
            }
          />
        </label>
      </div>
      <div className="segment-meta">
        <span>Range: {firstFrame} to {lastFrame}</span>
        <span>{metadata?.frame_count ?? 0} frames</span>
      </div>
    </div>
  );
}
