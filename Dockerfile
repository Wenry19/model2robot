# ============================================================
# Base image
# ROS 2 Humble Desktop on Ubuntu 22.04
# ============================================================

FROM osrf/ros:humble-desktop-jammy

# ============================================================
# System dependencies
# ============================================================

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# PyTorch
# ============================================================

RUN pip3 install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121

# ============================================================
# TensorRT
# ============================================================

RUN pip3 install --no-cache-dir \
    --extra-index-url https://pypi.nvidia.com \
    tensorrt-cu12==10.7.0

# ============================================================
# Python dependencies
# ============================================================

RUN pip3 install --no-cache-dir \
    onnx==1.12.0 \
    pandas==1.3.5 \
    pillow==10.4.0 \
    scikit-learn==1.0.2 \
    cuda-bindings==12.8.0\
    nvidia-ml-py==12.535.108

# ============================================================
# Upgrade pip
# ============================================================

RUN pip3 install --no-cache-dir --upgrade pip

# ============================================================
# Python Import Path
# ============================================================

ENV PYTHONPATH=/model2robot:${PYTHONPATH}
