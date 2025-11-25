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
cd ..

python train_tau_agent.py --llm-proxy --model Qwen/Qwen3-4B-Instruct-2507 --n-gpus 4 2>&1 | tee agent_train.log