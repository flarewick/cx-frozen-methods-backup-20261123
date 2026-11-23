#!/usr/bin/env bash
set -euo pipefail
RT=/data/cj/robo_pi/250010033/qfw/robotwin
cd /opt/cx_deployment
python -m pip install --no-cache-dir --no-deps torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
# This is a replay of the audited installed inventory, not a fresh resolver
# solution to the old openpi training pyproject. Source overlays follow below.
python -m pip install --no-cache-dir --no-deps -r requirements.runtime.txt
python -m pip install --no-cache-dir 'hatchling==1.27.0' 'editables==0.5'
python install_overlay.py
python -m pip install --no-deps --no-build-isolation -e "$RT/RoboTwin/policy/pi05"
export MAX_JOBS="${MAX_JOBS:-2}" SETUPTOOLS_SCM_PRETEND_VERSION=0.7.7.post1.dev5
export TORCH_CUDA_ARCH_LIST=8.9 CUDA_HOME=/usr/local/cuda
# Source snapshot excludes all old .so/.o/build files. Compile for RTX 4090.
python -m pip install --no-deps --no-build-isolation -e "$RT/RoboTwin/envs/curobo"
ln -sfnT /opt/cx_venv "$RT/venv_pi05_rt"
ln -sfnT /usr/local/cuda "$RT/cuda_unified"
ln -sfnT "$RT/checkpoints" "$RT/RoboTwin/policy/pi05/checkpoints"
mkdir -p "$RT/bin"
# Host driver libraries are supplied by NVIDIA Container Toolkit. Never import
# the source host's extracted 580.x driver or its absolute Vulkan ICD path.
cat > "$RT/setup_user_nvidia_graphics.env" <<'ENV'
export __GLX_VENDOR_LIBRARY_NAME=nvidia
ENV
python doctor.py --cpu
