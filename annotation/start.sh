#!/bin/bash

# Script to start the Clinical Decision Annotation System

echo "Clinical Decision Annotation System"
echo "===================================="
echo ""

# Activate conda environment
echo "Activating conda environment 'alignment'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate alignment

# Create annotations directory if it doesn't exist
if [ ! -d "annotations" ]; then
    echo "Creating annotations directory..."
    mkdir -p annotations
    echo ""
fi

# Check if data file exists
DATA_FILE="../experiments/cache/agent_deepseek.json"
if [ ! -f "$DATA_FILE" ]; then
    echo "WARNING: Data file not found at $DATA_FILE"
    echo "Please ensure the data file exists before starting the application."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start the Flask application
echo "Starting Flask application..."
echo "Access the application at: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py
