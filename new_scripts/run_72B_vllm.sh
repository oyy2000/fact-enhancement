#!/bin/bash
# Run Qwen2.5-72B-Instruct with 8-GPU tensor parallelism via vLLM
# Requires all 8 GPUs to be free (~144GB model in fp16)
set -e

cd /common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_CACHE_ROOT=/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/new_exps/.cache/vllm
export XDG_CACHE_HOME=/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/new_exps/.cache
export HF_HOME=/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/new_exps/.cache/huggingface
export TRANSFORMERS_CACHE=/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/new_exps/.cache/huggingface/hub

/common/home/sl2148/anaconda3/envs/fact_yang/bin/python new_scripts/figure1_sampling_vllm.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor_parallel_size 8 \
    --num_samples 8 \
    --temperature 0.7 \
    --max_tokens 2048 \
2>&1 | tee new_exps/figure1_sampling_data/log_72B_vllm.txt
