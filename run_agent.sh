#!/bin/bash


set +x
# SFT + GRPO
# 当前时间
TIME=$(date +%Y%m%d_%H%M%S)


cp PIKE_RAG/env_configs/.env_data PIKE_RAG/env_configs/.env

bash start_eval_model.sh 0,1,2,3,4,5,6,7 8 Qwen/Qwen3-4B-Instruct-2507 true
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
sleep 30
python agent.py exp_name=data_collection_gptnew seed=$1 task=search mode=train iterations=5  | tee agent_train.log
sleep infinity