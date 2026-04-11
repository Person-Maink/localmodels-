#!/usr/bin/env python3
import subprocess
import sys

ARGS = [sys.executable, '/home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/analysis/3D Visualization /Camera.py', '--frames_root', 'None', '--vipe_pose_file', '/home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/outputs/vipe/pose/120-2_clip_4.npz', '--camera_poses_file', 'None']
raise SystemExit(subprocess.call(ARGS))
