#!/bin/bash
# Queue Qwen2.5-72B-Instruct inference on GPUs 4-7
# Polls every 30s until all 4 GPUs have >= FREE_THRESHOLD MiB free
# Then launches vLLM with tensor_parallel_size=4

set -e

PYTHON=/common/home/sl2148/anaconda3/envs/fact_yang/bin/python
BASE=/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement
SCRIPT=$BASE/new_scripts/figure1_sampling_vllm.py
LOGDIR=$BASE/new_exps/figure1_sampling_data
POLL_INTERVAL=30
FREE_THRESHOLD=39000  # MiB; ~38GB free means GPU is essentially empty
TARGET_GPUS="4 5 6 7"
TARGET_GPUS_CSV="4,5,6,7"

cd $BASE

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_CACHE_ROOT=$BASE/new_exps/.cache/vllm
export XDG_CACHE_HOME=$BASE/new_exps/.cache
export HF_HOME=$BASE/new_exps/.cache/huggingface
export TRANSFORMERS_CACHE=$BASE/new_exps/.cache/huggingface/hub

mkdir -p $LOGDIR

wait_for_gpus() {
    echo "=========================================="
    echo "Waiting for GPUs $TARGET_GPUS_CSV to have >= ${FREE_THRESHOLD} MiB free..."
    echo "Polling every ${POLL_INTERVAL}s. Press Ctrl+C to abort."
    echo "=========================================="

    while true; do
        all_ready=true
        status_line=""
        for gpu in $TARGET_GPUS; do
            free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu 2>/dev/null | tr -d ' ')
            used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu 2>/dev/null | tr -d ' ')
            status_line="$status_line GPU$gpu:${free_mib}free/${used_mib}used"
            if [ "$free_mib" -lt "$FREE_THRESHOLD" ] 2>/dev/null; then
                all_ready=false
            fi
        done

        ts=$(date '+%Y-%m-%d %H:%M:%S')
        if [ "$all_ready" = true ]; then
            echo "[$ts] ALL READY!$status_line"
            return 0
        else
            echo "[$ts] Waiting...$status_line"
            sleep $POLL_INTERVAL
        fi
    done
}

echo "=== Qwen2.5-72B-Instruct Sampling Queue ==="
echo "Start time: $(date)"
echo "Target GPUs: $TARGET_GPUS_CSV"
echo "Free threshold: ${FREE_THRESHOLD} MiB per GPU"
echo ""

# Check if 72B data already exists and is complete
OUTFILE=$LOGDIR/Qwen_Qwen2.5-72B-Instruct/gsm8k_samples.jsonl
if [ -f "$OUTFILE" ]; then
    lines=$(wc -l < "$OUTFILE")
    if [ "$lines" -ge 1319 ]; then
        echo "72B data already complete ($lines lines). Exiting."
        exit 0
    else
        echo "72B data incomplete ($lines/1319 lines). Will re-run."
    fi
fi

wait_for_gpus

echo ""
echo "=========================================="
echo "Launching Qwen2.5-72B-Instruct at $(date)"
echo "GPUs: $TARGET_GPUS_CSV, Tensor Parallel: 4"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$TARGET_GPUS_CSV $PYTHON $SCRIPT \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor_parallel_size 4 \
    --num_samples 8 \
    --temperature 0.7 \
    --max_tokens 2048 \
    --gpu_memory_utilization 0.95 \
    --max_model_len 4096 \
    2>&1 | tee $LOGDIR/log_72B_vllm_v3.txt

EXIT_CODE=$?
echo ""
echo "=== 72B finished at $(date) with exit code $EXIT_CODE ==="

if [ $EXIT_CODE -ne 0 ]; then
    echo "72B FAILED! Check $LOGDIR/log_72B_vllm_v3.txt for details."
    echo ""
    echo "If OOM, consider:"
    echo "  1. Using --max_model_len 2048 to reduce KV cache"
    echo "  2. Using all 8 GPUs: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 --tensor_parallel_size 8"
    exit 1
fi

# Verify output
if [ -f "$OUTFILE" ]; then
    lines=$(wc -l < "$OUTFILE")
    echo "Output: $OUTFILE ($lines lines)"
    $PYTHON -c "
import json
total, correct = 0, 0
with open('$OUTFILE') as f:
    for line in f:
        obj = json.loads(line)
        for s in obj['samples']:
            total += 1
            if s['correct']:
                correct += 1
print(f'Total samples: {total}, Correct: {correct}, pass@1: {correct/total:.3f}')
"
fi
