devices=$1
tensor_parallel_size=$2
echo "Starting Baichuan API on devices: $devices with tensor parallel size: $tensor_parallel_size"

if $4; then
    uv venv vllmenv -p 3.10 --clear
    source vllmenv/bin/activate
    # uv pip install -U vllm==0.9.0
    uv pip install -U "ray[data, llm, serve]==2.49.2"
    uv pip install -U "click==8.2.1"
    uv pip install -U transformers
else
    source vllmenv/bin/activate
fi
# uv pip install -U "transformers!=4.52.0,<=4.52.4,>=4.49.0"
CUDA_VISIBLE_DEVICES=$devices vllm serve $3 --trust-remote-code --tensor-parallel-size $tensor_parallel_size --gpu-memory-utilization 0.8 --port 9000 &
# sleep 120
# bash start_eval_model.sh 4,5,6,7 4 zai-org/GLM-4.7 true