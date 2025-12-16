cd PIKE_RAG
source .venv/bin/activate
python agent_data_traj.py task=rare_disease iter=0
python agent_data_sft.py task=rare_disease iter=0 debug=false
cd ..