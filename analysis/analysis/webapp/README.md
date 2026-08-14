# Tremor Web App

This workspace contains a FastAPI backend and a React + TypeScript frontend for the tremor video analysis project.

## Layout

- `backend/`
  - Filesystem catalog scan for videos, sources, and ViPE overlays
  - DynHAMR normalization hook that uses the existing repo conversion logic
  - Explicit mode registry derived from the current visualization and analysis scripts
  - Visualization scene generation and frequency-analysis execution
- `frontend/`
  - Shared sidebar source browser with `By Clip`, `By Source`, and `By Experiment`
  - `Visualization` and `Frequency Analysis` tabs
  - Dynamic mode-control rendering from backend metadata
  - `vtk.js` scene canvas and Plotly result views

## Backend

From the repo root:

```bash
uv sync
.venv/bin/python -m uvicorn webapp.backend.main:app --reload
```

Useful endpoints:

- `GET /api/library/tree`
- `POST /api/library/rescan`
- `GET /api/modes/visualization`
- `GET /api/modes/analysis`
- `POST /api/visualization/manifest`
- `POST /api/visualization/frames`
- `POST /api/analysis/run`

## Frontend

From `webapp/frontend`:

```bash
npm install
npm run dev
```

Set `VITE_API_ROOT` if the backend is not running at `http://127.0.0.1:8000`.

## Notes

- The web app does not use `FILENAME.py`.
- WHIM is intentionally excluded.
- ViPE is treated as an auxiliary visualization overlay only.
- DynHAMR discovery is normalization-first, then filesystem scan.
- Plot exports are transient; the app does not persist analysis images into the repo.
