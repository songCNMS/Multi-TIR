# Setup Search API environment
conda create -n retriever python=3.10 -y
source ~/anaconda3/etc/profile.d/conda.sh
conda activate retriever
conda install -y pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini omegaconf
conda install -y -c pytorch -c nvidia faiss-gpu=1.8.0
pip install uvicorn fastapi tqdm
cd Search_R1/search_r1/search
bash build_index_symptoms.sh
bash build_index.sh
cd ../../..
cd Search_R1
bash retrieval_launch.sh &
bash retrieval_launch_symptoms.sh &
bash retrieval_logprobs_launch.sh &
sleep infinity