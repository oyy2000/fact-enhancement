#!/bin/bash
# Run remaining 6 tasks sequentially on GPU 0-3, TP=4
# No pipe/tee to avoid CUDA fork issues
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRANSFORMERS_CACHE=/common/users/sl2148/.cache/huggingface
export HF_HOME=/common/users/sl2148/.cache/huggingface

PYTHON=/common/home/sl2148/anaconda3/envs/fact_yang/bin/python
SCRIPT=/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/new_scripts/figure1_multi_dataset_vllm.py
LOGDIR=/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/new_exps/figure1_multi_dataset/logs
MAINLOG="${LOGDIR}/e4_remaining.log"

COMMON_ARGS="--tensor_parallel_size 4 --num_samples 16 --temperature 0.7 --max_tokens 2048 --gpu_memory_utilization 0.92 --max_model_len 2048 --swap_space 64 --enforce_eager"

TASKS=(
    "meta-llama/Llama-3.1-70B-Instruct math500"
    "meta-llama/Llama-3.1-70B-Instruct olympiad"
    "Qwen/Qwen2.5-72B-Instruct math500"
    "Qwen/Qwen2.5-72B-Instruct aime"
    "Qwen/Qwen2.5-72B-Instruct amc"
    "Qwen/Qwen2.5-72B-Instruct olympiad"
)

echo "$(date) Starting 6 remaining tasks on GPU 0-3 (TP=4)" >> "$MAINLOG"
echo "Params: enforce_eager, gpu_mem=0.92, max_model_len=2048, swap=64" >> "$MAINLOG"

for task in "${TASKS[@]}"; do
    MODEL=$(echo $task | cut -d' ' -f1)
    DATASET=$(echo $task | cut -d' ' -f2)
    SHORT=$(echo $MODEL | sed 's|.*/||' | sed 's/-Instruct//')
    LOG="${LOGDIR}/${DATASET}_${SHORT}.log"

    echo "$(date) [START] ${SHORT} x ${DATASET}" >> "$MAINLOG"
    
    $PYTHON -u $SCRIPT \
        --model $MODEL \
        --dataset $DATASET \
        $COMMON_ARGS \
        > "$LOG" 2>&1
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "$(date) [DONE] ${SHORT} x ${DATASET}" >> "$MAINLOG"
    else
        echo "$(date) [FAIL] ${SHORT} x ${DATASET} (exit=$EXIT_CODE)" >> "$MAINLOG"
    fi
done

echo "$(date) All tasks finished." >> "$MAINLOG"
