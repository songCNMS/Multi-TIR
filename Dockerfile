# Use NVIDIA CUDA base image matching the environment requirement (CUDA 12.6, Ubuntu 22.04)
# Note: The original yaml specified 'amlt-sing/acpt-torch2.7.1-py3.10-cuda12.6-ubuntu22.04'
# We use a public equivalent base and install dependencies manually.
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# Remove interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# Combined from job_search_agent.yaml setup and run_agl_dist.sh requirements
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    curl \
    git \
    libmpich-dev \
    libopenmpi-dev \
    g++ \
    g++-aarch64-linux-gnu \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js (required for agent-lightning/dashboard)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs

# Create python symlinks
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Install uv
RUN pip install uv

# Set environment variables from yaml
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
ENV PATH=/root/.local/bin:$PATH
# PYTHONPATH will be updated when activating venv or custom setting

# Setup CUDA symlink as requested
RUN ln -s /usr/local/cuda/lib64/libcudart.so /usr/lib/libcudart.so || true

# Set working directory
WORKDIR /app

# Copy the agent-lightning directory first to handle dependencies
COPY agent-lightning ./agent-lightning

# Build dashboard and install dependencies with uv
# Logic taken from run_agl_dist.sh
WORKDIR /app/agent-lightning/dashboard
RUN npm ci && npm run build

WORKDIR /app/agent-lightning
# Sync uv dependencies
# Note: --frozen requires uv.lock to exist. If it doesn't, remove --frozen.
# Assuming uv.lock exists in source.
# Enforce Python 3.10 to match faiss-gpu requirements and avoiding uv downloading newer python versions
RUN uv sync \
    --python 3.10 \
    --extra apo \
    --extra verl \
    --group dev \
    --group torch-gpu-stable \
    --group trl \
    --group agents \
    --no-default-groups

# Copy the PIKE_RAG directory
WORKDIR /app
COPY PIKE_RAG ./PIKE_RAG

# Activate virtual environment and install PIKE_RAG requirements and other packages
# We concatenate commands to ensure the venv is active for the installs
RUN . ./agent-lightning/.venv/bin/activate && \
    uv pip install -r PIKE_RAG/requirements.txt && \
    uv pip install vllm==0.10.2 && \
    uv pip install verl==0.5.0 && \
    uv pip install flash-attn --no-build-isolation && \
    uv pip install click==8.2.1 && \
    # Note: torch==2.8.0 is not yet a standard public release. 
    # If this fails, consider using a stable version or a specific index-url.
    uv pip install -U torch==2.8.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu121 || true && \
    uv pip install jsonlines && \
    uv pip install faiss-gpu && \
    uv pip install -U "numpy<2.0"

# Copy environment config
RUN cp PIKE_RAG/env_configs/.env_amlt PIKE_RAG/env_configs/.env

# Set the working directory to PIKE_RAG for execution
WORKDIR /app/PIKE_RAG

# Create entrypoint script
# We bypass start_retriever.sh to use the image's environment and avoid re-installation
# We replicate the server start command here.
RUN echo '#!/bin/bash\n\
set -e\n\
source ../agent-lightning/.venv/bin/activate\n\
\n\
# Retriever configuration\n\
RETRIEVER_INDEX_PATH=${RETRIEVER_INDEX_PATH:-/mnt/storage/data/search/data_output/retriever_index}\n\
RETRIEVER_NAME=${RETRIEVER_NAME:-e5}\n\
RETRIEVER_MODEL=${RETRIEVER_MODEL:-intfloat/e5-base-v2}\n\
\n\
echo "Starting Retriever Server..."\n\
python ./search_r1/retrieval_server.py \\\n\
    --index_path "$RETRIEVER_INDEX_PATH/e5_Flat.index" \\\n\
    --corpus_path "$RETRIEVER_INDEX_PATH/wiki-18.jsonl" \\\n\
    --topk 3 \\\n\
    --retriever_name "$RETRIEVER_NAME" \\\n\
    --retriever_model "$RETRIEVER_MODEL" &\n\
\n\
# Wait for server/process to initialize if needed (optional)\n\
sleep 5\n\
\n\
# Default values for env vars if not set\n\
export FINAL_ANSWER_MODEL=${FINAL_ANSWER_MODEL:-gpt}\n\
export TOOL_CALL_DENSE_REWARD_ON=${TOOL_CALL_DENSE_REWARD_ON:-false}\n\
export DENSE_REWARD_ON=${DENSE_REWARD_ON:-false}\n\
\n\
echo "Running train_search_agent.py with args: $@"\n\
exec python train_search_agent.py --llm-proxy "$@"\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# # Environment variables for execution
# ENV NCCL_SHM_DISABLE=1
# ENV NCCL_P2P_DISABLE=1

# Expose ports if necessary
EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--help"]


# docker build -t search-agent .
# docker run --gpus all \
#   -v /mnt/storage:/mnt/storage \
#   -e RETRIEVER_INDEX_PATH=/mnt/storage/data/search/data_output/retriever_index \
#   -e FINAL_ANSWER_MODEL=gpt \
#   search-agent \
#   --model Qwen/Qwen2.5-7B --n-gpus 8 --exp-name my_experiment