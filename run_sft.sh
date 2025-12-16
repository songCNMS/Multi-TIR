base_model=Qwen/Qwen3-4B-Instruct-2507
r=16
alpha=16
method=$2
# method=sft
cd LLaMA-Factory
echo "Setting up LLaMA-Factory SFT environment..."
if $1 ; then
    uv venv -p 3.10 --clear
    source .venv/bin/activate
    uv pip install -r requirements.txt 
    uv pip install wandb datasets flashinfer-python langchain_huggingface rouge_score jsonlines bottle omegaconf huggingface_hub[hf_xet]
    uv pip install vllm
    uv pip install "deepspeed>=0.10.0,<=0.16.9"
    uv pip install -e ".[torch,metrics]"
    uv pip install omegaconf
else
    source .venv/bin/activate
fi
echo "Preparing SFT data..."
python prepare_sft_data.py file_loc=/mnt/storage/data/rare_disease/data_output/data_sft_0.jsonl
echo "Starting LLaMA-Factory SFT training..."
python config_generate.py model=$base_model r=$r alpha=$alpha
llamafactory-cli train llama-factory/$base_model-$method-lora-r$r-alpha$alpha.yaml
echo "Exporting the trained model..."
llamafactory-cli export llama-factory/$base_model-$method-lora-r$r-alpha$alpha-merge.yaml