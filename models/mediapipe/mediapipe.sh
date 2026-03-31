#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "Use ${SCRIPT_DIR}/inference.sh for manual Mediapipe SLURM runs."
echo "Submitting that script keeps Mediapipe aligned with the other model launchers."
