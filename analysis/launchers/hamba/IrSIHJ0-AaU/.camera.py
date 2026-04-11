#!/usr/bin/env python3
import subprocess
import sys

ARGS = [sys.executable, '/home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/analysis/3D Visualization /Camera.py', '--frames_root', '/home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/outputs/hamba/IrSIHJ0-AaU/meshes', '--vipe_pose_file', 'None', '--camera_poses_file', 'None']
raise SystemExit(subprocess.call(ARGS))
