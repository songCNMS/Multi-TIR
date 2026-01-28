# Base image from the YAML configuration
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04


# Set working directory
WORKDIR /workspace

# Environment variables
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV PATH=${HOME}/.local/bin:${PATH}
ENV PYTHONPATH=${PWD}:${PYTHONPATH}
ENV NCCL_SHM_DISABLE=1
ENV NCCL_P2P_DISABLE=1
ENV HF_CACHE_DIR=/mnt/storage/data/huggingface
ENV WANDB_API_KEY=""
ENV HF_TOKEN=""

# System setup and dependencies
RUN apt-get update -y && \
    ln -s /usr/local/cuda/lib64/libcudart.so /usr/lib/libcudart.so || true && \
    apt install -y libmpich-dev g++-aarch64-linux-gnu libopenmpi-dev g++ npm && \
    curl -sL https://aka.ms/InstallAzureCLIDeb | bash && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Accept conda TOS
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Install uv package manager
RUN pip install uv

# Copy the project files
COPY . /workspace/

# Clone tau2-bench and set up data
RUN git clone https://github.com/sierra-research/tau2-bench.git && \
    mkdir -p PIKE_RAG/tau2_bench/ && \
    cp -r tau2-bench/data PIKE_RAG/tau2_bench/

# Build agent-lightning dashboard
WORKDIR /workspace/agent-lightning/dashboard
RUN npm ci && npm run build

# Install Python dependencies with uv
WORKDIR /workspace/agent-lightning
RUN uv sync --frozen \
    --extra apo \
    --extra verl \
    --group dev \
    --group torch-gpu-stable \
    --group trl \
    --group agents \
    --no-default-groups

# Install additional Python packages
RUN . /workspace/agent-lightning/.venv/bin/activate && \
    uv pip install -r /workspace/PIKE_RAG/requirements.txt && \
    uv pip install vllm==0.10.2 && \
    uv pip install verl==0.5.0 && \
    uv pip install flash-attn --no-build-isolation && \
    uv pip install click==8.2.1 && \
    uv pip install -U torch==2.8.0 torchvision && \
    uv pip install "numpy<2.0"

# Install tau2_bench
WORKDIR /workspace/PIKE_RAG/tau2_bench
RUN . /workspace/agent-lightning/.venv/bin/activate && \
    uv pip install -e .

# Set final working directory
WORKDIR /workspace/PIKE_RAG

# Activate virtual environment in shell sessions
RUN echo "source /workspace/agent-lightning/.venv/bin/activate" >> ~/.bashrc

# Default command - activate venv and keep container running
CMD ["/bin/bash", "-c", "source /workspace/agent-lightning/.venv/bin/activate && bash"]


# docker build -t tau-training:latest .
# docker run --gpus all -it \
#   -e WANDB_API_KEY="your_key" \
#   -e HF_TOKEN="your_token" \
#   -v /path/to/storage:/mnt/storage \
#   tau-training:latest