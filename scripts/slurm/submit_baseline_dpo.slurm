#!/bin/bash
#SBATCH --job-name=baseline-dpo
#SBATCH --account=bdhh-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=16g
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Baseline agent inference + principal inference with Llama-DPO

# 1. Initialize Shell and Environment
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate sglang

# 2. Start the SGLang Server in the Background
echo "Starting SGLang server on port 30000..."
bash interface/sglang.sh DPO &
SERVER_PID=$!

# 3. Wait for the Server to be Ready
echo "Waiting for server to initialize..."
while ! nc -z localhost 30000; do
  sleep 5
done
echo "Server is up and listening on port 30000."

# 4. Run Baseline Inference (agent + principal)
echo "Starting baseline inference script..."
conda deactivate
conda activate alignment

# Args: agent_model dataset inference_mode
# Override MAX_CASES at submission time: MAX_CASES=0 sbatch ... (0 = no limit)
AGENT_SERVER=sglang MAX_CASES="${MAX_CASES:-100}" bash scripts/run_baseline.sh llama-dpo usmle_sample full

# 5. Cleanup
echo "Inference complete. Shutting down server."
kill $SERVER_PID
