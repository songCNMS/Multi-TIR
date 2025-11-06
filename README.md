Env: CUDA12.8

**Step 0 - Env Setup**
- Start all tools server
- bash start_retriever_API.sh 
- bash start_baichuan_API.sh 0,1,2,3 4 # change to your devices accordingly

**Step 1 - Training**
- Data collection: run python agent.py mode=train exp_name=data_collection seed=42 task=rare_disease
- Transfer to SFT data: 
- Run SFT: 
- Transfer to single-turn RL:
- Run RL:
- Inference with a fine-tune model: 

**Step 2 - Multi-turn RL**