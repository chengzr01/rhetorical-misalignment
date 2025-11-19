#!/bin/bash
#SBATCH --job-name=sglang-inference
#SBATCH --account=bdhh-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=16g
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out  # Saves output to logs/jobname-jobid.out
#SBATCH --error=logs/%x-%j.err

# 1. Initialize Shell and Environment
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate alignment

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

# 4. Run Inference
# IMPORTANT: Ensure run_inference.sh targets port 30000, not 30001
echo "Starting inference script..."
bash scripts/run_inference.sh llama-sft usmle_sample full

# 5. Cleanup
# Once inference is done, we kill the server process
echo "Inference complete. Shutting down server."
kill $SERVER_PID