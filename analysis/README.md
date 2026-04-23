# Comparative Study Visualization Toolkit

This project contains a set of scripts for generating visualizations from hand-tracking and hand-mesh outputs. The visualizations fall into three broad groups:

- 3D scene visualizations
- frequency analysis plots
- helper launchers and montage generation

Most scripts are meant to be run directly with Python. Some also have generated launchers under `launchers/` for per-clip convenience.

## Visualization Types

### 1. 3D Visualizations

These scripts render hand geometry, camera motion, or related scene elements in a 3D viewer.

- `3D Visualization /Camera.py`
  - Visualizes ViPE and frame-camera trajectories in one scene.
  - Useful when comparing camera motion between sources.
  - Supports configurable frame inputs, ViPE pose input, camera pose input, and frame globs.

- `3D Visualization /Free.py`
  - Renders a free-view 3D hand visualization.
  - Useful for inspecting the hand shape without a fixed camera framing.
  - Source is injected through `FREE_SOURCE` in `FILENAME.py` or via generated launchers.

- `3D Visualization /Wrist Grounding.py`
  - Shows wrist-grounded hand geometry.
  - Useful for understanding motion relative to the wrist origin.
  - Source is injected through `WRIST_GROUNDING_SOURCE` in `FILENAME.py` or via generated launchers.

- `3D Visualization /Bounding Boxes.py`
  - Visualizes WiLoR bounding-box tracks from per-frame outputs.
  - Useful for checking box tracking behavior over time.
  - Supports frame directory globs and subsampling controls.

- `3D Visualization /MANO.py`
  - Visualizes MANO face and vertex indices.
  - Useful as a geometry/debugging helper rather than a clip-based visualization.

### 2. WHIM Visualizations

These scripts work on WHIM per-video annotation directories.

- `3D Visualization /WHIM Camera.py`
  - Visualizes WHIM camera tracks.
- `3D Visualization /WHIM Free.py`
  - Free-view WHIM mesh visualization.
- `3D Visualization /WHIM Bounding Boxes.py`
  - Visualizes WHIM bounding-box tracks.
- `3D Visualization /WHIM.py`
  - Combined WHIM visualization, if present in the workspace.

The WHIM scripts all take `--video-dir` and operate on a per-video annotation folder.

### 3. Frequency Analysis Plots

These scripts generate time-series and frequency-domain plots for hand motion signals.

- `Frequency Analysis/Compare.py`
  - Compares multiple sources in one figure.
  - Plots displacement over time, power spectral density, and per-axis filtered displacement.
  - Default mode compares two sources.
  - `--all-models` compares WiLoR, Hamba, DynHAMR, and MediaPipe together.

- `Frequency Analysis/Point to Point.py`
  - Measures motion between two points or regions.
  - Plots filtered region-difference displacement over time, frequency spectrum, and per-axis displacement.
  - Default mode compares two sources.
  - `--all-models` compares all configured model families together.

- `Frequency Analysis/Multi Point to Point.py`
  - Extends point-to-point analysis to multiple point pairs at once.
  - Useful when you want one figure with several pairwise trajectories for the same clip.
  - Supports MANO pairs and MediaPipe pairs.

- `Frequency Analysis/Run All.py`
  - Batch runner that discovers all configured sources, runs the frequency analyses, and saves figures plus CSV/JSON summaries.
  - Produces SVG output for plots.
  - Supports:
    - `--scenario` for filtering the scenario matrix
    - `--all-models` for multi-model jobs
    - `--only-missing` to skip jobs whose outputs already exist
    - `--max-pairs` to cap the workload
    - `--dry-run` to inspect discovery without running analyses

## Ways To Generate Visualizations

### Direct script execution

Run a script directly with Python when you want a single visualization or an interactive viewer.

Examples:

```bash
python3 "Frequency Analysis/Compare.py"
python3 "Frequency Analysis/Point to Point.py" --all-models
python3 "Frequency Analysis/Multi Point to Point.py"
python3 "3D Visualization /Camera.py"
```

### Per-clip launchers

The `generate_visualization_launchers.py` script creates a tree of executable launcher scripts under `launchers/`.

Each launcher corresponds to a specific clip and visualization type, for example:

- `launchers/wilor/<clip>/camera`
- `launchers/wilor/<clip>/free`
- `launchers/wilor/<clip>/wrist_grounding`
- `launchers/wilor/<clip>/bounding_boxes`
- `launchers/wilor_finetune/<experiment>/<clip>/camera`
- `launchers/wilor_finetune/<experiment>/<clip>/free`
- `launchers/wilor_finetune/<experiment>/<clip>/wrist_grounding`
- `launchers/vipe/<clip>/camera`
- `launchers/mediapipe/<clip>/free`
- `launchers/whim_train/<clip>/camera`

These launchers are the easiest way to open the same visualization repeatedly for different clips.

### Batch analysis output

Use `Frequency Analysis/Run All.py` when you want:

- a full set of comparison plots,
- saved artifacts for later review,
- CSV metrics,
- JSON summaries.

This is the best option when you want many plots generated in one pass.

### README montage generation

`make_readme_gif.py` builds a GIF montage from existing video and analysis outputs.

It is not a plotting script itself, but it helps package the project’s results into a single README-friendly asset.

## Output Locations

Common output locations are controlled by `FILENAME.py`.

- `CONFIG.OUTPUTS_ROOT` for analysis/model outputs
- `CONFIG.ANALYSIS_OUTPUT_DIR` for frequency-analysis artifacts
- `launchers/` for generated shortcut scripts

The frequency-analysis runner writes:

- plot files as SVG
- `metrics.csv`
- `summary.json`

## Data Sources

The visualization scripts can work with several source types:

- WiLoR mesh/frame outputs
- WiLoR finetune mesh/frame outputs
- Hamba mesh/frame outputs
- DynHAMR mesh/frame outputs
- ViPE pose outputs
- MediaPipe CSV keypoint outputs
- WHIM per-video annotation directories

## Configuration

Most paths and defaults come from `FILENAME.py`. That file is the main place to configure:

- input roots
- default clip selections
- analysis parameters
- frequency-analysis source choices

Some scripts also accept command-line overrides for source paths, frame sampling, and visualization settings.

## Practical Notes

- 3D visualization scripts usually require `vedo`.
- Frequency-analysis scripts use Matplotlib and save plots headlessly.
- SVG is the preferred plot export format for analysis figures.
- If a script says a source is `None`, it usually means the corresponding path has not been configured in `FILENAME.py`.

## Suggested Starting Points

If you want to explore the project quickly:

1. Use `generate_visualization_launchers.py` to build clip-specific launchers.
2. Use `Frequency Analysis/Run All.py` to generate batch SVG plots and summaries.
3. Use one of the `3D Visualization /...` scripts for an interactive view of a single clip.
