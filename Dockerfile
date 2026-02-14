FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system packages required for building and running Python image processing code
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-setuptools build-essential cmake git wget \
    ffmpeg libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 libv4l-dev \
    libjpeg-dev libpng-dev libtiff-dev pkg-config ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Make python and pip point to python3
RUN ln -sf /usr/bin/python3 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

RUN pip install --upgrade pip wheel setuptools

# Install CUDA-compatible PyTorch + torchvision (adjust versions if you need different CUDA)
# This example uses the official PyTorch cu118 wheels. Change if you target another CUDA version.
RUN pip install --no-cache-dir torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118

# Install Python requirements (requirements.txt is copied first to leverage Docker layer caching)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt || true

# Copy project into the image
WORKDIR /app
COPY . /app

# Default command (override at runtime). For headless use you can change this or override in docker run/compose.
CMD ["python", "D01_Llamacpp_v1.py"]
