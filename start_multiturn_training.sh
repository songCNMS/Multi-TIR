# host=localhost
# port=8010
# tool_type=search_retrieval_rare,search_retrieval_rare_cases,search_retrieval_brare_feedack # separate by comma if you want to start multiple tool servers
# workers_per_tool=8 # number of workers for the tool server, meaning how many threads will be used to handle a single tool request with multiple trajectories
# python -m verl_tool.servers.serve --host $host --port $port --tool_type $tool_type --workers_per_tool $workers_per_tool # run in background


train_data="/mnt/storage/data/rare_disease/logs/rare/train_tool.parquet"
val_data="/mnt/storage/data/rare_disease/logs/rare/test_tool.parquet"

export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH



# bash start_tool_server_single.sh local Qwen/Qwen2.5-3B-Instruct 1 4 16 16 4 no

model_name=$2
total_training_steps=10000 # total training steps, set to 1000000 for long-term training
# model_name=/mnt/storage/data/rare_disease/logs/sft/Qwen/Qwen2.5-3B-Instruct-r32-alpha16
rl_alg=grpo # gae(ppo) or grpo, if grpo, then better set n>1 otherwise the group norm can not be effective
n_gpus_per_node=$4
n_nodes=$3
lora_rank=$5
lora_alpha=$6
tensor_model_parallel_size=$7
create_env=$8

if [ $create_env == "yes" ]; then
    echo "Creating conda environment verl-tool-env..."
    pip install uv
    uv venv -p 3.10 --clear
    source .venv/bin/activate
    uv pip install rouge_score langchain_huggingface jsonlines omegaconf wikipedia flask
    uv pip install -e verl
    uv pip install -e ".[vllm,acecoder,torl,search_tool]"
    # uv pip install "flash-attn<2.8.0" --no-build-isolation
    uv pip install -U ray
    uv pip install -U vllm==0.9.0
    # uv pip install -U torch torchvision # --index-url https://download.pytorch.org/whl/cu130
    uv pip install -U https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.11/flash_attn-2.8.3+cu126torch2.7-cp310-cp310-linux_x86_64.whl
    uv pip install -U torch==2.7 torchvision --index-url https://download.pytorch.org/whl/cu126
    # uv pip install -U https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.11/flash_attn-2.8.3+cu129torch2.8-cp310-cp310-linux_x86_64.whl
    # https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.18/flash_attn-2.8.3+cu130torch2.9-cp310-cp310-linux_x86_64.whl
    uv pip install -U "transformers<4.54.0"
    uv pip install "triton==3.1.0"
    uv pip install "click==8.2.1"
else
    echo "Skipping conda environment creation."
    source .venv/bin/activate
fi


n=8
batch_size=32
ppo_mini_batch_size=1
max_prompt_length=10000
max_num_batched_tokens=15000
max_response_length=2048
max_obs_length=2048
temperature=1.0
top_p=1.0
strategy="fsdp"
action_stop_tokens='</tool_call>'
additional_eos_token_ids=[151645]
max_turns=3
kl_loss_coef=0.0
kl_coef=0
entropy_coeff=0
kl_loss_type=low_var_kl
lr=1e-6
reward_manager=torl
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1
gpu_memory_utilization=0.5 # higher gpu_memory_utilization will likely cause the vllm to OOM and get stuck, so set it to a lower value like 0.4 or 0.5
do_offload=True # control actor's fsdp.[param|optimizer]_offload and actor_rollout_ref.rollout.fsdp.[param|optimizer]_offload; if gpu_memory_utilization is set to > 0.6, then do_offload should be set to True otherwise it will cause OOM
use_dynamic_bsz=True # faster
ulysses_sequence_parallel_size=1 # set to 1 for normal verl behavior, otherwise it will cause OOM
fsdp_size=-1
mask_observations=True # mask observations for kl loss and gradient descent
enable_mtrl=True # enable multi-turn training
max_action_length=$((max_response_length*max_turns))

# sudo ln -s /usr/include /usr/lib/include

export GLOO_SOCKET_IFNAME=ibs10f1
export NCCL_SOCKET_IFNAME=ibs10f1
export VLLM_USE_V1=1

ray stop
ray start --head --node-ip-address=$(hostname -I | awk '{print $2}') --port=6379 --include-dashboard false
sleep 10

export MEDICAL_LLM_ENDPOINT="http://10.150.240.105:8000/v1"
export MEDICAL_LLM_KEY="token-abc123"
export RARE_CASE_SEARCH_ENDPOINT="http://10.150.240.105:8020/retrieve"
export RARE_DISEASE_SEARCH_ENDPOINT="http://10.150.240.105:8010/retrieve"

# temp file for action tokens as verl cannot pass special strs as params
mkdir -p $(pwd)/tmp
action_stop_tokens_file="$(pwd)/tmp/action_stop_tokens.txt"
echo -e -n "$action_stop_tokens" | tee $action_stop_tokens_file
echo "action_stop_tokens_file=$action_stop_tokens_file"

model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
run_name_postfix="-mtrl-v6"
run_name="${reward_manager}-${strategy}-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-t${temperature}-lr${lr}${run_name_postfix}"
export VERL_RUN_ID=$run_name
export NCCL_DEBUG=INFO

echo "Starting tool server..."
host=$(hostname -I | awk '{print $2}')
port=$(shuf -i 30000-31000 -n 1)
# port=8010
# search_retrieval_rare_feedback,search_retrieval_rare_diagnosis
tool_server_url=http://$host:$port/get_observation
python -m verl_tool.servers.serve --host $host --port $port --tool_type "get_rare_disease_information_local,get_similar_cases,get_rare_disease_information_from_wikipedia" --workers_per_tool 8 &
server_pid=$!

sleep 20

echo "Server (pid=$server_pid) started at $tool_server_url"
# python examples/data_preprocess/rare_tool.py 

PYTHONUNBUFFERED=1 python -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.val_batch_size=$batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation='right' \
    reward_model.reward_manager=$reward_manager \
    custom_reward_function.path=examples/data_preprocess/rare_reward_tool.py \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length + 1024)) \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.lora_rank=${lora_rank} \
    actor_rollout_ref.model.lora_alpha=${lora_alpha} \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    actor_rollout_ref.agent.max_response_length=$max_response_length \
    actor_rollout_ref.agent.max_start_length=$max_prompt_length \
    actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    actor_rollout_ref.agent.max_turns=$max_turns \
    actor_rollout_ref.agent.mask_observations=$mask_observations \
    actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
    actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
    actor_rollout_ref.agent.max_action_length=$max_action_length \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length + 2048)) \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.load_format="safetensors" \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    critic.optim.lr=1e-5 \
    critic.strategy=$strategy \
    critic.model.path=$model_name \
    critic.model.fsdp_config.fsdp_size=$fsdp_size \
    critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    critic.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$reward_manager \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    trainer.save_freq=$total_training_steps \
    trainer.test_freq=$total_training_steps \
    trainer.total_training_steps=$total_training_steps
pkill -P -9 $server_pid
kill -9 $kill $server_pid


