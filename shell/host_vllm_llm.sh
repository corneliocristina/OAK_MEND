#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn

llm=${1:-"Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"}
port=${2:-"9110"}
tensor_parallel_size=${3:-"2"}
max_model_len=${4:-"20480"}
max_num_batched_tokens=${5:-"131072"}
max_num_seq=${6:-"32"}

vllm serve "${llm}" --port ${port} \
  --gpu-memory-utilization 0.94 --enable-prefix-caching --enable-chunked-prefill \
  --max-model-len ${max_model_len} \
  --max-num-seqs ${max_num_seq} \
  --max-num-batched-tokens ${max_num_batched_tokens} \
  --tensor-parallel-size ${tensor_parallel_size}
