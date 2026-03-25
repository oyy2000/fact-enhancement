#!/bin/bash
# Queue Qwen2.5-32B and 72B inference on GPUs 4-7
# Polls until GPUs have enough free memory, then launches.
# 32B: needs 2 GPUs (~64GB fp16), 72B: needs 4 GPUs (~144GB fp16)

set -euo pipefail

PYTHON=/common/home/sl2148/anaconda3/envs/fact_yang/bin/python
BASE=/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement
SCRIPT=$BASE/new_scripts/figure1_sampling_vllm.py
LOGDIR=$BASE/new_exps/figure1_sampling_data
POLL_INTERVAL=30
FREE_THRESHOLD=39000  # MiB; GPUs must be essentially empty

cd $BASE

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_CACHE_ROOT=$BASE/new_exps/.cache/vllm
export XDG_CACHE_HOME=$BASE/new_exps/.cache
export HF_HOME=$BASE/new_exps/.cache/huggingface
export TRANSFORMERS_CACHE=$BASE/new_exps/.cache/huggingface/hub

mkdir -p $LOGDIR

check_gpu_free() {
    local gpu_id=$1
    local free_mib
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | tr -d ' ')
    echo "$free_mib"
}

wait_for_n_gpus() {
    local needed=$1
    shift
    local gpu_list=("$@")
    local desc="need $needed free GPUs from [${gpu_list[*]}]"

    echo "=========================================="
    echo "Waiting for $needed GPUs to have >= ${FREE_THRESHOLD} MiB free"
    echo "Candidates: ${gpu_list[*]}"
    echo "Polling every ${POLL_INTERVAL}s..."
    echo "=========================================="

    while true; do
        local ready_gpus=()
        local status_line=""

        for gpu in "${gpu_list[@]}"; do
            local free_mib
            free_mib=$(check_gpu_free "$gpu")
            local used_mib
            used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ')
            status_line="$status_line GPU$gpu:${free_mib}f/${used_mib}u"
            if [ "$free_mib" -ge "$FREE_THRESHOLD" ] 2>/dev/null; then
                ready_gpus+=("$gpu")
            fi
        done

        local ts
        ts=$(date '+%m-%d %H:%M:%S')

        if [ "${#ready_gpus[@]}" -ge "$needed" ]; then
            local selected="${ready_gpus[*]:0:$needed}"
            echo "[$ts] READY! ${#ready_gpus[@]}/$needed GPUs free.$status_line"
            echo "Selected GPUs: $selected"
            SELECTED_GPUS=$(echo "$selected" | tr ' ' ',')
            return 0
        else
            echo "[$ts] ${#ready_gpus[@]}/$needed ready.$status_line"
            sleep $POLL_INTERVAL
        fi
    done
}

verify_output() {
    local outfile=$1
    local model_name=$2
    if [ -f "$outfile" ]; then
        local lines
        lines=$(wc -l < "$outfile")
        if [ "$lines" -ge 1319 ]; then
            echo "$model_name: COMPLETE ($lines lines)"
            $PYTHON -c "
import json
total, correct = 0, 0
with open('$outfile') as f:
    for line in f:
        obj = json.loads(line)
        for s in obj['samples']:
            total += 1
            if s['correct']:
                correct += 1
print(f'  Total samples: {total}, Correct: {correct}, pass@1: {correct/total:.3f}')
"
            return 0
        fi
    fi
    return 1
}

echo "==========================================="
echo "  32B + 72B Sampling Queue"
echo "  Start: $(date)"
echo "  GPUs: 4,5,6,7"
echo "  Threshold: ${FREE_THRESHOLD} MiB free/GPU"
echo "==========================================="
echo ""

# ============================================
# STEP 1: Qwen2.5-32B-Instruct (2 GPUs)
# ============================================
OUT_32B=$LOGDIR/Qwen_Qwen2.5-32B-Instruct/gsm8k_samples.jsonl

echo "=== [1/2] Qwen2.5-32B-Instruct ==="
if verify_output "$OUT_32B" "32B"; then
    echo "Skipping 32B (already complete)."
else
    echo "32B data missing or incomplete. Queuing..."
    SELECTED_GPUS=""
    wait_for_n_gpus 2 4 5 6 7

    echo ""
    echo "Launching 32B on GPUs $SELECTED_GPUS at $(date)"
    TP_SIZE=$(echo "$SELECTED_GPUS" | tr ',' '\n' | wc -l)

    CUDA_VISIBLE_DEVICES=$SELECTED_GPUS $PYTHON $SCRIPT \
        --model Qwen/Qwen2.5-32B-Instruct \
        --tensor_parallel_size "$TP_SIZE" \
        --num_samples 8 \
        --temperature 0.7 \
        --max_tokens 2048 \
        --gpu_memory_utilization 0.92 \
        --max_model_len 4096 \
        2>&1 | tee $LOGDIR/log_32B_vllm_q.txt

    echo "32B finished at $(date)"
    verify_output "$OUT_32B" "32B" || echo "WARNING: 32B output incomplete!"
fi

echo ""

# ============================================
# STEP 2: Qwen2.5-72B-Instruct (4 GPUs)
# ============================================
OUT_72B=$LOGDIR/Qwen_Qwen2.5-72B-Instruct/gsm8k_samples.jsonl

echo "=== [2/2] Qwen2.5-72B-Instruct ==="
if verify_output "$OUT_72B" "72B"; then
    echo "Skipping 72B (already complete)."
else
    echo "72B data missing or incomplete. Queuing..."
    SELECTED_GPUS=""
    wait_for_n_gpus 4 4 5 6 7

    echo ""
    echo "Launching 72B on GPUs $SELECTED_GPUS at $(date)"

    CUDA_VISIBLE_DEVICES=$SELECTED_GPUS $PYTHON $SCRIPT \
        --model Qwen/Qwen2.5-72B-Instruct \
        --tensor_parallel_size 4 \
        --num_samples 8 \
        --temperature 0.7 \
        --max_tokens 2048 \
        --gpu_memory_utilization 0.95 \
        --max_model_len 4096 \
        2>&1 | tee $LOGDIR/log_72B_vllm_q.txt

    EXIT_CODE=${PIPESTATUS[0]}
    echo "72B finished at $(date) (exit=$EXIT_CODE)"

    if [ "$EXIT_CODE" -ne 0 ]; then
        echo ""
        echo "72B FAILED. Retrying with max_model_len=2048..."
        SELECTED_GPUS=""
        wait_for_n_gpus 4 4 5 6 7

        CUDA_VISIBLE_DEVICES=$SELECTED_GPUS $PYTHON $SCRIPT \
            --model Qwen/Qwen2.5-72B-Instruct \
            --tensor_parallel_size 4 \
            --num_samples 8 \
            --temperature 0.7 \
            --max_tokens 2048 \
            --gpu_memory_utilization 0.97 \
            --max_model_len 2048 \
            --enforce_eager \
            2>&1 | tee $LOGDIR/log_72B_vllm_q_retry.txt

        echo "72B retry finished at $(date)"
    fi

    verify_output "$OUT_72B" "72B" || echo "WARNING: 72B output incomplete!"
fi

echo ""
echo "==========================================="
echo "  ALL DONE at $(date)"
echo "==========================================="
