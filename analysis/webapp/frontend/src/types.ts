export type CapabilityFlags = {
  visualization: boolean;
  frequency: boolean;
  camera: boolean;
  bounding_boxes: boolean;
  beta: boolean;
  auxiliary_only: boolean;
};

export type SourceAsset = {
  id: string;
  family: string;
  label: string;
  clip_id: string;
  family_id: string;
  path: string;
  experiment?: string | null;
  video_id?: string | null;
  capabilities: CapabilityFlags;
};

export type VideoAsset = {
  id: string;
  label: string;
  stem: string;
  path: string;
};

export type OverlayAsset = {
  id: string;
  kind: string;
  clip_id: string;
  family_id: string;
  pose_path: string;
  rgb_video_id?: string | null;
};

export type TreeViewPayload = {
  scanned_at: string;
  organization_modes: string[];
  sources: Record<string, SourceAsset>;
  videos: Record<string, VideoAsset>;
  overlays: Record<string, OverlayAsset>;
  views: {
    by_clip: any[];
    by_source: any[];
    by_experiment: any[];
  };
};

export type ControlSpec = {
  id: string;
  label: string;
  type: "number" | "boolean" | "select" | "text";
  required?: boolean;
  default?: unknown;
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
  help?: string | null;
  options?: { label: string; value: string }[];
};

export type ModeSpec = {
  id: string;
  label: string;
  tab: "visualization" | "analysis";
  script_name: string;
  purpose: string;
  required_inputs: string[];
  optional_arguments: string[];
  expected_outputs: string[];
  ui_appearance: string;
  min_sources: number;
  max_sources?: number | null;
  supported_families: string[];
  controls: ControlSpec[];
};

export type SceneActor = {
  id: string;
  kind: "mesh" | "points" | "segments";
  label: string;
  source_id?: string | null;
  hand?: "left" | "right" | "unknown" | null;
  color: string;
  opacity: number;
  points: number[][];
  faces: number[][];
  segments: number[][][];
  meta?: Record<string, unknown>;
};

export type FrameScene = {
  frame_id: number;
  actors: SceneActor[];
};

export type VisualizationManifest = {
  request_id: string;
  mode_id: string;
  fps: number;
  frame_ids: number[];
  source_colors: Record<string, string>;
  available_hands: string[];
  static_actors: SceneActor[];
  camera_display: Record<string, unknown>;
};

export type VisualizationFrames = {
  request_id: string;
  mode_id: string;
  scenes: FrameScene[];
};

export type SourceMetadata = {
  source_id: string;
  frame_count: number;
  frame_ids: number[];
  first_frame_id?: number | null;
  last_frame_id?: number | null;
  available_hands: string[];
  bounding_boxes: boolean;
  path: string;
};

export type AnalysisJob = {
  job_id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  result_id?: string | null;
  error?: string | null;
};

export type AnalysisResultEntry = {
  label: string;
  source_id?: string;
  pair_label?: string;
  hand_used?: string;
  dominant_hz?: number;
  rms_amplitude?: number;
  sample_count?: number;
  plots?: {
    time_s: number[];
    magnitude: number[];
    freqs_hz: number[];
    psd: number[];
    filtered_xyz: number[][];
  };
  series?: {
    point_count: number;
    dominant_hz: number;
    rms_amplitude: number;
    sample_count: number;
  }[];
};

export type AnalysisResult = {
  result_id: string;
  mode_id: string;
  title: string;
  inputs: Record<string, unknown>;
  entries: AnalysisResultEntry[];
  tables: Record<string, unknown>[];
  metrics: Record<string, unknown>;
  figure_svg_url?: string | null;
};
