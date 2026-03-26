# Video-Based Tremor Analysis

<p align="center">
  <img src="assets/readme/me1_pipeline_montage.gif" alt="Custom me 1 montage showing the input video, WiLoR output, and a grounded WiLoR analysis view in 3D space" width="82%" />
</p>

This repository explores whether clinically relevant tremor characteristics can be estimated from ordinary video rather than relying only on specialized electrophysiological registration. The current codebase is organized as a comparative research workspace: multiple video and hand-motion pipelines are run on the same clips, their outputs are normalized into a common folder structure, and downstream analysis scripts compare rhythmicity, dominant frequency, and motion amplitude.

The project is motivated by tremor assessment in neuromuscular disorders, but the repository reflects the more recent implementation work rather than an older thesis description. In practice, this repo is less a single model and more a full research workflow for benchmarking video-based motion analysis methods.

## What This Repo Does

- Uses `models/vipe` to estimate camera pose and intrinsics artifacts from video.
- Uses `models/wilor_hands` and `models/hamba` to reconstruct hand motion and export comparable per-frame outputs.
- Uses `models/mediapipe` as a lightweight landmark baseline.
- Uses `analysis/` to compare outputs across methods and derive frequency- and amplitude-oriented metrics from the recovered motion.

## Current Pipeline

1. **Input videos / extracted frames** live under `data/`.
2. **Geometry + camera estimation** is produced with [`models/vipe`](models/vipe/README.md).
3. **Hand reconstruction / tracking baselines** are run with [`models/wilor_hands`](models/wilor_hands), [`models/hamba`](models/hamba/README.md), [`models/mediapipe`](models/mediapipe), and optionally [`models/dyn-hamr`](models/dyn-hamr/README.md).
4. **Frequency and motion analysis** is performed in [`analysis/`](analysis), especially the scripts in [`analysis/Frequency Analysis/`](analysis/Frequency%20Analysis).

A typical artifact flow in this repo looks like:

`data/` -> `outputs/vipe/` -> `outputs/wilor/`, `outputs/hamba/`, `outputs/mediapipe/` -> `analysis/`

## Model Overview

| Folder | Role | Input | Main output | Notes |
| --- | --- | --- | --- | --- |
| `models/vipe` | Camera pose, intrinsics, depth-oriented geometry artifacts | Raw videos | `outputs/vipe/pose`, `outputs/vipe/intrinsics`, related artifacts | Geometry backbone for downstream comparison |
| `models/wilor_hands` | WiLoR-based hand reconstruction and fine-tuning | Extracted frames / frame folders | `outputs/wilor/<clip>/meshes`, `visualizations`, videos | Contains comparison wrappers plus WiLoR-specific training scripts |
| `models/hamba` | Hamba-based hand reconstruction baseline | Extracted frames / frame folders | `outputs/hamba/<clip>/meshes`, `visualizations`, videos | Adapted to match the repo's comparison-oriented I/O style |
| `models/mediapipe` | Lightweight landmark baseline | Videos | `outputs/mediapipe/keypoints`, `outputs/mediapipe/visualizations` | Useful baseline when full 3D reconstruction is unnecessary |
| `models/dyn-hamr` | Optional camera-aware dynamic hand motion reconstruction | Videos | `outputs/dynhamr/logs/...` | More specialized 4D method; useful as an additional comparison point |

## Repository Structure

```text
comparative study/
├── data/           # source videos, extracted frames, dataset inputs
├── models/         # model wrappers, training scripts, and upstream research code
├── analysis/       # comparison, frequency analysis, and visualization scripts
├── outputs/        # generated artifacts from each pipeline
└── other scripts/  # ad hoc utilities and experiments
```

## How To Navigate This Repo

The root README is the project index. Most model folders have their own setup details, dependencies, and upstream documentation:

- [`models/vipe/README.md`](models/vipe/README.md) for the camera and geometry pipeline.
- [`models/hamba/README.md`](models/hamba/README.md) for the Hamba baseline.
- [`models/dyn-hamr/README.md`](models/dyn-hamr/README.md) for Dyn-HaMR.
- [`models/wilor_hands/WiLoR/README.md`](models/wilor_hands/WiLoR/README.md) for the upstream WiLoR project used inside this workspace.

Not every folder under `models/` is original work from this repo. Several subdirectories vendor or wrap upstream research code. The main value of this repository is the **comparative layer** built around them: shared data conventions, output organization, model-specific wrappers, and downstream analysis for tremor-oriented evaluation.

## Typical Workflow

### 1. Run model-specific pipelines from `models/`

Examples of the wrapper-style entry points used in this repo:

```bash
cd models/wilor_hands
python main.py --image_folder ../../data/images --output_folder ../../outputs/wilor
```

```bash
cd models/hamba
python main.py --image_folder ../../data/images --output_folder ../../outputs/hamba
```

```bash
cd models/mediapipe
python main.py --video_folder ../../data/images --output_folder ../../outputs/mediapipe/
```

For ViPE and Dyn-HaMR, use their own READMEs and run scripts as the source of truth for environment setup and execution.

### 2. Store generated artifacts under `outputs/`

This repo already follows a method-specific output layout:

- `outputs/vipe/` for pose, intrinsics, depth, masks, and related camera artifacts.
- `outputs/wilor/` for WiLoR-based reconstruction results.
- `outputs/hamba/` for Hamba-based reconstruction results.
- `outputs/mediapipe/` for keypoints and overlay videos.
- `outputs/dynhamr/` for Dyn-HaMR fitting logs and exports.

### 3. Run downstream comparative analysis

The analysis layer is where the repo becomes a tremor-analysis workspace rather than a collection of unrelated model folders.

```bash
cd analysis
python "Frequency Analysis/Run All.py" --output-dir ../outputs/analysis_images
```

Related scripts under [`analysis/Frequency Analysis/`](analysis/Frequency%20Analysis) compare model outputs and generate plots/metrics for dominant frequency, spectral content, and amplitude-oriented motion summaries.

## Notes And Limitations

- This is an exploratory research repository, not a clinical product.
- The code investigates whether video can recover tremor-relevant information, but it does **not** claim to replace standard electrophysiological tremor registration in clinical practice.
- Different model folders have different environments, dependencies, and runtime assumptions. The root README is therefore an index and workflow guide, not a promise of a single unified installation path.
- Some command paths and scripts are cluster- or container-oriented, so subproject READMEs should be treated as the authoritative setup reference for each model.

## Why This Repo Exists

The central question behind this workspace is not just whether a single model can track a hand, but whether a video-based pipeline can produce motion signals stable enough for tremor-oriented analysis. That is why the repository combines:

- multiple reconstruction baselines,
- camera-aware geometry artifacts,
- shared output conventions,
- and downstream analysis scripts that compare the resulting motion signals rather than only the visual overlays.

In other words, this repo is designed to support **comparison, interpretation, and analysis**, not only inference.
