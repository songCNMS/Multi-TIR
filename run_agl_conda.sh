git clone https://github.com/sierra-research/tau2-bench.git
cp -r tau2-bench/data PIKE_RAG/tau2_bench/
cd agent-lightning
cd dashboard
npm ci
npm run build
cd ..
pip install uv
conda create -n tauenv python=3.12 -y
conda activate tauenv
conda install -c conda-forge libstdcxx-ng=14.1 -y
pip install -r PIKE_RAG/requirements.txt
pip install -U agentlightning[verl]
pip install vllm==0.10.2
pip install verl==0.5.0
pip install flash-attn --no-build-isolation
pip install click==8.2.1
pip install -U torch==2.8.0 torchvision
cd PIKE_RAG/tau2_bench
pip install -e .
pip install "numpy<2.0"
cd ../..
export PYTHONPATH=$PWD/agent-lightning:$PYTHONPATH
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
cd PIKE_RAG
# python train_tau_agent.py --llm-proxy --model /mnt/storage/data/tau/logs/sft-tool-calling/Qwen/Qwen3-4B-Instruct-2507-r16-alpha16
# /mnt/storage/data/tau/logs/sft-tool-calling/Qwen/Qwen3-4B-Instruct-2507-r16-alpha16 --n-gpus 2 --exp-name sft-tc 2>&1 | tee agent_train.log

# python train_tau_agent.py --llm-proxy --model Qwen/Qwen3-4B-Instruct-2507 --exp-name hindsight_tc 2>&1 | tee agent_train.log

python train_tau_agent.py --llm-proxy --model $1 --exp-name $2 2>&1 | tee agent_train.log