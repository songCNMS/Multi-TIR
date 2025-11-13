cd PIKE_RAG
source .venv/bin/activate
python agent_data_traj.py task=rare_disease
python agent_data_sft.py task=rare_disease iter=0
cd ..