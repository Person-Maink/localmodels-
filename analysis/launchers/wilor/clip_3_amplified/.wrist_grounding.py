#!/usr/bin/env python3
import importlib.util
import sys
from pathlib import Path

ANALYSIS_ROOT = '/home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/analysis'
VIS_ROOT = '/home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/analysis/3D Visualization '
for path in (VIS_ROOT, ANALYSIS_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import FILENAME as CONFIG
CONFIG.WRIST_GROUNDING_SOURCE = Path('/home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/outputs/wilor/clip_3_amplified/meshes')

spec = importlib.util.spec_from_file_location('wrist_grounding_launcher_target', '/home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/analysis/3D Visualization /Wrist Grounding.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
raise SystemExit(module.main())
