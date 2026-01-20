
if $5; then
    echo "Setting up Agent-Lightning environment..."
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
else
    source agent-lightning/.venv/bin/activate
fi

cd PIKE_RAG
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1

CUDA_VISIBLE_DEVICES=$3 python train_math_agent.py --llm-proxy --model $1 --n-gpus $2 --exp-name $4 2>&1 | tee agent_train.log

# python train_math_agent.py --llm-proxy --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --n-gpus 4 --exp-name math_train  2>&1 | tee agent_train.log