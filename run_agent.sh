seed=$1
cd PIKE_RAG
uv venv -p 3.10 --clear
source .venv/bin/activate
uv pip install -r requirements.txt
python prompt_templates.py
python agent.py exp_name=data_collection seed=$seed task=rare_disease mode=train iterations=10 cfg_loc=$2
cd ..