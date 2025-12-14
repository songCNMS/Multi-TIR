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

export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1

# python train_tau_agent.py --llm-proxy --model /mnt/storage/data/tau/logs/sft-tool-calling/Qwen/Qwen3-4B-Instruct-2507-r16-alpha16 --n-gpus 2 --exp-name sft-tc 2>&1 | tee agent_train.log

python train_tau_agent.py --llm-proxy --model Qwen/Qwen3-4B-Instruct-2507 --n-gpus 2 --exp-name origin-nontc 2>&1 | tee agent_train.log

python train_tau_agent.py --llm-proxy --model $1 --n-gpus $2 --exp-name $3 2>&1 | tee agent_train.log