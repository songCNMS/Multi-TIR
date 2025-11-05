devices=$1
tensor_parallel_size=$2
uv venv vllmenv -p 3.10 --clear
source vllmenv/bin/activate
uv pip install -U vllm==0.9.0
uv pip install -U "transformers!=4.52.0,<=4.52.4,>=4.49.0"
CUDA_VISIBLE_DEVICES=$devices vllm serve baichuan-inc/Baichuan-M2-32B --trust-remote-code --tensor-parallel-size $tensor_parallel_size --gpu-memory-utilization 0.95