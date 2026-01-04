cd PIKE_RAG
uv venv -p 3.10 --clear
source .venv/bin/activate
uv pip install -r requirements.txt 
python agent_data_traj.py task=search iter=0
python agent_data_sft.py task=search iter=0
cd ..