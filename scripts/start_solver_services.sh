#!/bin/bash

model_path=$1

if [ -z "$model_path" ]; then
    echo "Error: model_path is required"
    echo "Usage: bash start_solver_services.sh <model_path>"
    exit 1
fi

echo "Starting 1 solver service with model: $model_path"

mkdir -p ${STORAGE_PATH:-/tmp}/temp_results

export VLLM_DISABLE_COMPILE_CACHE=1

CUDA_VISIBLE_DEVICES=0,5 python vllm_service_init/start_vllm_server.py --port 5000 --model_path "$model_path" --gpu_mem_util 0.9 --tensor_parallel_size 2 > solver_service.log 2>&1 &
solver_pid=$!

echo "Solver service PID: $solver_pid"

echo "Solver service started on port 5000"
echo "Waiting for services to initialize..."
sleep 30

echo "Solver services ready for questioner training"
exit 0