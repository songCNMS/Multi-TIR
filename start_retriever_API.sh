# Setup Search API environment
# conda create -n retriever python=3.10 -y
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate retriever
if $2; then
    echo "Setting up Retriever API environment..."
    uv venv retrieverenv -p 3.10 --clear
    source retrieverenv/bin/activate
    conda install -y pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
    uv pip install transformers datasets pyserini omegaconf
    # conda install -y -c pytorch -c nvidia faiss-gpu=1.8.0
    # conda install -c conda-forge faiss-gpu
    uv pip install faiss-cpu
    uv pip install accelerate
    uv pip install uvicorn fastapi tqdm
    uv pip install -U "numpy<2.0"
    cd Search_R1/search_r1/search
    bash build_index_symptoms.sh
    bash build_index.sh
    cd ../../..
else
    source retrieverenv/bin/activate
fi
cd Search_R1
echo "Launching Retriever API servers..."
bash retrieval_launch.sh $1 &
bash retrieval_launch_symptoms.sh $1 &
# bash retrieval_logprobs_launch.sh &
# sleep 120