#!/bin/bash
# Configuration parameters for the HMP-capable WiLoR Apptainer container.
# This file mirrors params.sh so the original image setup stays untouched.

export NAME=template-hmp

export STAGE_PATH="dblue:/home/${USER}/ondemand/jupyter/"
export DEPLOY_PATH="daic:/tudelft.net/staff-umbrella/reit/apptainer/"

if [[ -z "${NAME}" ]]; then
    echo "Error: NAME must not be empty" >&2
    return 1 2>/dev/null || exit 1
fi

if [[ -z "${STAGE_PATH}" ]]; then
    echo "Warning: STAGE_PATH is not set in params_hmp.sh" >&2
fi

if [[ -z "${DEPLOY_PATH}" ]]; then
    echo "Warning: DEPLOY_PATH is not set in params_hmp.sh" >&2
fi
