#!/bin/bash


set +x
# SFT + GRPO
# 当前时间
TIME=$(date +%Y%m%d_%H%M%S)

export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
HEAD_IP=$MASTER_ADDR
LOCAL_IP=$(hostname | awk '{print $1}')  # Get the local IP address
PORT=$MASTER_PORT
echo "MASTER_ADDR: $MASTER_ADDR:$MASTER_PORT"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# export RAY_ADDRESS="$HEAD_IP:$PORT"

mv PIKE_RAG/env_configs/.env_amlt PIKE_RAG/env_configs/.env

if [ "$LOCAL_IP" == "$HEAD_IP" ]; then
    cd agent-lightning
    cd dashboard
    npm ci
    npm run build
    cd ..
    pip install uv
    uv sync --frozen \
        --extra apo \
        --extra verl \
        --group dev \
        --group torch-gpu-stable \
        --group trl \
        --group agents \
        --no-default-groups
    cd ..
    source agent-lightning/.venv/bin/activate
    uv pip install -r PIKE_RAG/requirements.txt
    # uv pip install -U agentlightning[verl]
    uv pip install vllm==0.10.2
    uv pip install verl==0.5.0
    uv pip install flash-attn --no-build-isolation
    uv pip install click==8.2.1
    uv pip install -U torch==2.8.0 torchvision
    uv pip install jsonlines
    uv pip install -U "numpy<2.0"
    cd PIKE_RAG
    bash search_r1/start_retriever.sh true
    python train_search_agent.py --llm-proxy --model $1 --n-gpus $2 --dense-reward-on 2>&1 | tee agent_train.log
    sleep infinity
else
    cd PIKE_RAG; bash start_summary_api.sh
    sleep infinity
fi