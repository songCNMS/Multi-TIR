#!/bin/bash
set +x
# 当前时间
TIME=$(date +%Y%m%d_%H%M%S)

if $3; then
    git clone https://github.com/sierra-research/tau2-bench.git
    cp -r tau2-bench/data PIKE_RAG/tau2_bench/
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
    cd PIKE_RAG/tau2_bench
    uv pip install -e .
    uv pip install "numpy<2.0"
    cd ..
else
    source agent-lightning/.venv/bin/activate
    cd PIKE_RAG
fi
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1

export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
HEAD_IP=$MASTER_ADDR
LOCAL_IP=$(hostname | awk '{print $1}')  # Get the local IP address
PORT=$MASTER_PORT
echo "MASTER_ADDR: $MASTER_ADDR:$MASTER_PORT"
export RAY_ADDRESS="$HEAD_IP:$PORT"



# python train_tau_agent.py --llm-proxy --model /mnt/storage/data/tau/logs/sft-tool-calling/Qwen/Qwen3-4B-Instruct-2507-r16-alpha16
# /mnt/storage/data/tau/logs/sft-tool-calling/Qwen/Qwen3-4B-Instruct-2507-r16-alpha16 --n-gpus 2 --exp-name sft-tc 2>&1 | tee agent_train.log
# python train_tau_agent.py --llm-proxy --model Qwen/Qwen3-4B-Instruct-2507 --n-gpus 4 --exp-name hindsight_tc 2>&1 | tee agent_train.log

if [ "$LOCAL_IP" == "$HEAD_IP" ]; then
    echo "Local IP $LOCAL_IP ..."
    ray start --head --port=$PORT
else
    echo "Local IP $LOCAL_IP is Worker node, connecting to Head node $HEAD_IP..."
    ray start --address=$HEAD_IP:$PORT
fi

sleep 20

set -x



if [ "$LOCAL_IP" == "$HEAD_IP" ]; then
    export NCCL_DEBUG=INFO
    python train_tau_agent.py --llm-proxy --model $1 --exp-name $2 2>&1 | tee agent_train.log
fi