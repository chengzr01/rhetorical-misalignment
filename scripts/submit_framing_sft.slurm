#!/bin/bash
#SBATCH --job-name=framing-sft
#SBATCH --account=bdhh-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=16g
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Agent Presentation with Llama-SFT as the framing agent.
# Ground-truth source: factual_agg (aggregated factual claims from llama models)
#
# Override at submission time with environment variables, e.g.:
#   MAX_CASES=0 sbatch scripts/submit_framing_sft.slurm   (0 = no limit)

# 1. Initialize Shell and Environment
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate sglang

# 2. Start the SGLang Server in the Background
echo "Starting SGLang server on port 30000..."
bash interface/sglang.sh SFT &
SERVER_PID=$!

# 3. Wait for the Server to be Ready
echo "Waiting for server to initialize..."
while ! nc -z localhost 30000; do
  sleep 5
done
echo "Server is up and listening on port 30000."

# 4. Run Agent Presentation + Principal Inference
echo "Starting framing inference script..."
conda deactivate
conda activate alignment

export MAX_CASES="${MAX_CASES:-100}"

AGENT_SERVER=sglang bash scripts/run_framing.sh llama-sft usmle_sample factual_agg full

# 5. Cleanup
echo "Framing experiment complete. Shutting down server."
kill $SERVER_PID
