#!/bin/bash
#SBATCH --job-name=framing-sft
#SBATCH --account=bdhh-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=16g
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out  # Saves output to logs/jobname-jobid.out
#SBATCH --error=logs/%x-%j.err

# Agent Presentation with Llama-SFT model using DeepSeek as ground truth

# 1. Initialize Shell and Environment
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate sglang

# 2. Start the Server in the Background
echo "Starting SGLang server on port 30000..."
# We use '&' to push this to the background so the script continues
bash interface/sglang.sh SFT &
SERVER_PID=$!

# 3. Wait for the Server to be Ready
# We loop and check if port 30000 is listening before running inference
echo "Waiting for server to initialize..."
while ! nc -z localhost 30000; do
  sleep 5
done
echo "Server is up and listening on port 30000."

# 4. Run Agent Presentation + Principal Inference
echo "Starting framing inference script..."
conda deactivate
conda activate alignment

# Run with deepseek as ground truth
# Args: agent_model dataset ground_truth_model mode
bash scripts/run_framing.sh llama-sft usmle_sample deepseek full

# 5. Cleanup
# Once inference is done, we kill the server process
echo "Inference complete. Shutting down server."
kill $SERVER_PID
