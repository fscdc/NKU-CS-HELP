#!/bin/bash
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# Function to kill all background jobs started by this script
terminate() {
    echo "Terminating all background jobs..."
    jobs -p | xargs -r kill
    wait
    exit 1
}

# Set up trap to call terminate() function on script exit or interruption
trap terminate SIGINT SIGTERM


CUDA_VISIBLE_DEVICES=7 python main.py --n_epochs 10 \
    --model "cnn" \


CUDA_VISIBLE_DEVICES=7 python main.py --n_epochs 20 \
    --model "lstm" \