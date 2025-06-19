# Dockerfile for Dust3r (Conda + CUDA + RoPE compilation)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install base dependencies
RUN apt-get update && apt-get install -y \
    wget git build-essential cmake libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH="$CONDA_DIR/bin:$PATH"
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

# Create and activate dust3r environment with Python 3.11 and cmake
RUN conda create -n dust3r python=3.11 cmake=3.14.0 -y
SHELL ["/bin/bash", "-c"]
RUN echo "conda activate dust3r" >> ~/.bashrc
ENV CONDA_DEFAULT_ENV=dust3r

# Set working dir and clone repo
WORKDIR /workspace
RUN git clone --recursive https://github.com/naver/dust3r.git
WORKDIR /workspace/dust3r
RUN git submodule update --init --recursive

# Install core dependencies (with PyTorch and CUDA 12.1)
RUN conda install -n dust3r -c pytorch -c nvidia pytorch torchvision pytorch-cuda=12.1 -y
RUN /opt/conda/envs/dust3r/bin/pip install -r requirements.txt
RUN /opt/conda/envs/dust3r/bin/pip install -r requirements_optional.txt

# Compile CUDA RoPE kernels
WORKDIR /workspace/dust3r/croco/models/curope/
RUN /opt/conda/envs/dust3r/bin/python setup.py build_ext --inplace

WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]