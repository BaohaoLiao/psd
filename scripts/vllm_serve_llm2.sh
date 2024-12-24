#!/bin/bash

hostname --ip-address
MODEL="/mnt/nushare2/data/baliao/PLLMs/qwen/Qwen2.5-Math-1.5B-Instruct"
MODEL_NAME="Qwen2.5-Math-1.5B-Instruct"

export CUDA_VISIBLE_DEVICES=1
python -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --served-model-name $MODEL_NAME \
        --tensor-parallel-size 1 \
        --port 12342 \
        --host 0.0.0.0 \
        --trust-remote-code \
        --max-model-len 4096 \
        --enforce-eager \
        --enable_prefix_caching \
        --gpu_memory_utilization 0.9