#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn

sent_embed=${1:-"Qwen/Qwen3-Embedding-0.6B"}
port=${2:-"9200"}
max_model_len=4096
max_num_seq=512
max_num_batched_tokens=131072

vllm serve "${sent_embed}" --port ${port} --gpu-memory-utilization 0.94 \
  --max-model-len ${max_model_len} \
  --max-num-seqs ${max_num_seq} \
  --max-num-batched-tokens ${max_num_batched_tokens}
