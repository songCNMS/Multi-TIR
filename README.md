# Multi-TIR: Multi-Turn Interactive Retrieval Agent for Rare Disease Diagnosis

Multi-TIR is a framework for training and evaluating agents specialized in rare disease diagnosis. It leverages Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) with tool-augmented reasoning to improve diagnostic accuracy. The system integrates a retrieval engine, an LLM server, and an agent framework to simulate and solve medical cases.

## Prerequisites

- **OS:** Linux
- **CUDA:** 12.8
- **Hardware:** Multi-GPU setup recommended (e.g., 4-8 GPUs) for training and serving. If dense reward is on, extra GPUs would be required to host a summary LLM.
- **Package Managers:** `conda`, `uv`

## Project Structure

- `PIKE_RAG/`: Core agent implementation, RAG tools, and RL training scripts.
- `LLaMA-Factory/`: Framework for Supervised Fine-Tuning (SFT).
- `agent-lightning/`: RL framework (based on `verl`) for agent training.
- `Search_R1/`: Retrieval engine implementation.

## Quick Start

### Step 0: Environment Setup

Before running any training or agents, you need to start the necessary backend services (Retriever and LLM).

1.  **Configure Environment:**
    Check and update `PIKE_RAG/env_configs/.env` if necessary to match your local setup (ports, API keys, etc.).

2.  **Start Retriever API:**
    This script sets up the retrieval environment and launches the search API.
    ```bash
    bash start_retriever_API.sh <cuda_devices>
    # Example: bash start_retriever_API.sh 0,1,2,3
    ```

3.  **Start LLM API (Baichuan):**
    This script serves the Baichuan-M2-32B model using vLLM.
    ```bash
    bash start_baichuan_API.sh <cuda_devices> <tensor_parallel_size>
    # Example: bash start_baichuan_API.sh 0,1,2,3 4
    ```

### Step 1: Data Collection & SFT

1.  **Data Collection:**
    Run the agent to collect trajectories for training.
    ```bash
    cd PIKE_RAG
    python agent.py mode=train exp_name=data_collection seed=42 task=rare_disease
    cd ..
    ```

2.  **Run Supervised Fine-Tuning (SFT):**
    Fine-tune a base model (e.g., Qwen) using the collected data.
    ```bash
    # Standard SFT
    bash run_sft.sh true sft
    
    # SFT with Tool Calling support
    bash run_sft.sh true sft-tool-calling
    ```
    *   `true`: Sets up the LLaMA-Factory environment (set to `false` if already set up).
    *   `sft` / `sft-tool-calling`: The training method/recipe to use.

### Step 2: Multi-turn RL Training

Train the agent using Reinforcement Learning (GRPO) to improve multi-turn reasoning and tool use.

```bash
bash run_agl.sh <model_name> <num_gpus> <use_dense_reward> <env_suffix> <cuda_devices>
```

**Example:**
```bash
bash run_agl.sh Qwen/Qwen3-4B-Instruct-2507 4 false local 4,5,6,7
```

**Parameters:**
- `<model_name>`: The model to train (e.g., `Qwen/Qwen3-4B-Instruct-2507`).
- `<num_gpus>`: Number of GPUs to use for the RL training process.
- `<use_dense_reward>`: `true` or `false`. If `true`, enables dense reward calculation (requires summary API).
- `<env_suffix>`: Suffix for the `.env` file (e.g., `local` uses `.env_local`).
- `<cuda_devices>`: Specific CUDA device IDs to use (e.g., `4,5,6,7`).

## Notes

- Ensure all scripts are run from the root directory unless specified otherwise.
- Monitor logs in `PIKE_RAG/agent_train.log` and `sft.log` (if applicable) for training progress.
