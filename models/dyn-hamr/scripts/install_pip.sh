#!/usr/bin/env bash
set -e

# module load 2023r1-gcc11 cuda/11.7 openmpi/4.1.4 python/3.10.8 ninja

# DOESN'T WORK
# module load 2024r1 cuda/11.7 openmpi/4.1.6 py-torch/1.12.1 python/3.10.8 py-pip

# WORKS DA
# module load 2023r1 cuda/11.7 openmpi/4.1.4 py-torch/1.12.1 python/3.10.8 py-pip ninja

echo "Creating virtual environment"
python3.10 -m venv .dynhamr
echo "Activating virtual environment"

source $PWD/.dynhamr/bin/activate

# install pytorch
$PWD/.dynhamr/bin/pip install torch==1.13.0 torchvision==0.14.0 --index-url https://download.pytorch.org/whl/cu117

# torch-scatter
$PWD/.dynhamr/bin/pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

# install source
$PWD/.dynhamr/bin/pip install -e .

# install remaining requirements
$PWD/.dynhamr/bin/pip install -r requirements.txt

# install DROID-SLAM/DPVO
cd third-party/DROID-SLAM
../../.dynhamr/bin/python setup.py install
cd ../..

# install HaMeR
cd third-party/hamer
../../.dynhamr/bin/pip install -e .[all]
../../.dynhamr/bin/pip install -v -e third-party/ViTPose

# downgrade numpy
$PWD/.dynhamr/bin/pip install numpy==1.22.4

# install mano
cd ../manopth
../../.dynhamr/bin/pip install .

# fix some other brokwn packages
cd ../../
$PWD/.dynhamr/bin/pip install contourpy==1.0.7 matplotlib==3.5.3 opencv-python==4.6.0.66 scikit-image==0.19.3 scipy==1.8.1

# install the packages that are somehow missing idek
$PWD/.dynhamr/bin/pip install loguru arguments future consoleprinter plyfile

# make sure that the syspath of the file is updated??

echo "Make sure the correct requirements file is available"
# install human_body_priors
cd ../human_body_priors
../../.dynhamr/bin/pip install -r requirements.txt --no-deps
../../.dynhamr/bin/python setup.py develop

cd ../..


$PWD/.dynhamr/bin/pip install mediapipe==0.10.9  protobuf==3.20.3

# install pytorch AGAIN because it's overwritten???
$PWD/.dynhamr/bin/pip install torch==1.13.0 torchvision==0.14.0 --index-url https://download.pytorch.org/whl/cu117

# final hard pin so later installs cannot silently move NumPy/SciPy forward
$PWD/.dynhamr/bin/pip install --force-reinstall --no-deps \
    numpy==1.22.4 scipy==1.8.1 contourpy==1.0.7 matplotlib==3.5.3 \
    opencv-python==4.6.0.66 scikit-image==0.19.3 pandas==1.4.0

$PWD/.dynhamr/bin/python - <<'PY'
import numpy
import scipy
import trimesh
print("FINAL numpy", numpy.__version__)
print("FINAL scipy", scipy.__version__)
print("FINAL trimesh", trimesh.__version__)
PY
