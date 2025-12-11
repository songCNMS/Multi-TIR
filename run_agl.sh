# bash run_agl.sh Qwen/Qwen3-4B-Instruct-2507 4 true local 4,5,6,7

cp PIKE_RAG/env_configs/.env_$4 PIKE_RAG/env_configs/.env
bash start_retriever_API.sh 0,1,2,3

if $3; then
    cd PIKE_RAG
    echo "Dense reward is ON"
    bash start_summary_api.sh 0,1,2,3
    cd ..
else
    echo "Dense reward is OFF"
fi

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
CUDA_VISIBLE_DEVICES=$5 python train_rare_agent.py --llm-proxy --model $1 --n-gpus $2 --dense-reward-on 2>&1 | tee agent_train.log
sleep infinity