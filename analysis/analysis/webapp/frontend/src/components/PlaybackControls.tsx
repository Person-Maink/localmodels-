type PlaybackControlsProps = {
  playing: boolean;
  canPlay: boolean;
  currentFrameIndex: number;
  maxFrameIndex: number;
  currentFrameId?: number;
  fps?: number;
  onPlayPause: () => void;
  onStep: (delta: number) => void;
  onSeek: (nextIndex: number) => void;
};

export function PlaybackControls({
  playing,
  canPlay,
  currentFrameIndex,
  maxFrameIndex,
  currentFrameId,
  fps,
  onPlayPause,
  onStep,
  onSeek
}: PlaybackControlsProps) {
  return (
    <div className="playback-shell">
      <div className="playback-actions">
        <button disabled={!canPlay} onClick={() => onStep(-1)}>
          Back
        </button>
        <button disabled={!canPlay} onClick={onPlayPause}>
          {playing ? "Pause" : "Play"}
        </button>
        <button disabled={!canPlay} onClick={() => onStep(1)}>
          Forward
        </button>
      </div>
      <input
        className="timeline-slider"
        type="range"
        min={0}
        max={Math.max(0, maxFrameIndex)}
        step={1}
        value={currentFrameIndex}
        disabled={!canPlay}
        onChange={(event) => onSeek(Number(event.target.value))}
      />
      <div className="playback-meta">
        <span>Frame {currentFrameId ?? "-"}</span>
        <span>{fps ? `${fps.toFixed(2)} fps` : ""}</span>
      </div>
    </div>
  );
}
