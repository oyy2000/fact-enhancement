#!/bin/bash
# Re-run 14B on AIME, AMC, Olympiad with increased swap space.
# The 14B previously failed with "lack of CPU swap space" on these datasets.
set -e

cd /common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_CACHE_ROOT=/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/new_exps/.cache/vllm
export XDG_CACHE_HOME=/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/new_exps/.cache
export HF_HOME=/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/new_exps/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub

PY=/common/home/sl2148/anaconda3/envs/fact_yang/bin/python
SCRIPT=new_scripts/figure1_multi_dataset_vllm.py
LOGDIR=new_exps/figure1_multi_dataset/logs
mkdir -p $LOGDIR

MODEL=Qwen/Qwen2.5-14B-Instruct
GPU=4,5
TP=2

for DATASET in aime amc olympiad; do
    echo ""
    echo "=========================================="
    echo "  14B on $DATASET"
    echo "=========================================="
    CUDA_VISIBLE_DEVICES=$GPU $PY $SCRIPT \
        --model $MODEL \
        --dataset $DATASET \
        --num_samples 16 \
        --temperature 0.7 \
        --max_tokens 2048 \
        --tensor_parallel_size $TP \
        --max_model_len 4096 \
        --gpu_memory_utilization 0.95 \
        --swap_space 32 \
        2>&1 | tee "$LOGDIR/${DATASET}_Qwen2.5-14B_retry.log"
    echo "[$(date +%H:%M:%S)] Done: 14B on $DATASET"
done

echo ""
echo "All 14B re-runs complete!"
