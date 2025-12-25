cd PIKE_RAG
cp env_configs/.env_amlt env_configs/.env

bash search_r1/start_retriever.sh true


if $3; then
    echo "Dense reward is ON"
    bash start_summary_api.sh Qwen/Qwen3-4B-Instruct-2507 0,1 &
else
    echo "Dense reward is OFF"
fi

cd ..

if $4; then
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
# bash search_r1/start_retriever.sh true
# python train_search_agent.py --llm-proxy --model /mnt/storage/data/search/logs/sft-tool-calling/Qwen/Qwen3-4B-Instruct-2507-r16-alpha16 --n-gpus 8 2>&1 | tee agent_train.log
# python train_search_agent.py --llm-proxy --model Qwen/Qwen3-4B-Instruct-2507 --n-gpus 8 2>&1 | tee agent_train.log

if $3; then
    CUDA_VISIBLE_DEVICES=$6  DENSE_REWARD_ON=true python train_search_agent.py --llm-proxy --model $1 --n-gpus $2 --exp-name $5 2>&1 | tee agent_train.log
else
    CUDA_VISIBLE_DEVICES=$6  DENSE_REWARD_ON=false python train_search_agent.py --llm-proxy --model $1 --n-gpus $2 --exp-name $5 2>&1 | tee agent_train.log
fi

# CUDA_VISIBLE_DEVICES=0,1,2,3 DENSE_REWARD_ON=false python train_search_agent.py --llm-proxy --model Qwen/Qwen3-4B-Instruct-2507 --n-gpus 4 --exp-name b200-wot-sum-ds  2>&1 | tee agent_train.log