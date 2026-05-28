#!/bin/bash
# Comprehensive queue for ALL remaining Figure 1 / E4-E6 experiments.
#
# QWEN (already cached, runs immediately):
#   14B: aime, amc, olympiad
#   32B: math500, aime, amc, olympiad
#   72B: math500, aime, amc, olympiad         (11 tasks)
#
# LLAMA (needs HF auth - run after `huggingface-cli login`):
#   1B/3B/8B: all 5 datasets each             (15 tasks)
#   70B: all 5 datasets                        (5 tasks)
#
# Uses GPUs 4-7 with polling.

set -euo pipefail

PYTHON=/common/home/sl2148/anaconda3/envs/fact_yang/bin/python
BASE=/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement
GSM8K_SCRIPT=$BASE/new_scripts/figure1_sampling_vllm.py
MULTI_SCRIPT=$BASE/new_scripts/figure1_multi_dataset_vllm.py
LOGDIR=$BASE/new_exps/figure1_multi_dataset/logs
GSM8K_DATA=$BASE/new_exps/figure1_sampling_data
MULTI_DATA=$BASE/new_exps/figure1_multi_dataset
POLL_INTERVAL=30
FREE_THRESHOLD=39000

cd "$BASE"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_CACHE_ROOT=$BASE/new_exps/.cache/vllm
export XDG_CACHE_HOME=$BASE/new_exps/.cache
export HF_HOME=$BASE/new_exps/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub

# Pass HF token if available (needed for gated Llama models)
if [ -f "$HF_HOME/token" ]; then
    export HF_TOKEN=$(cat "$HF_HOME/token")
elif [ -f "/common/home/sl2148/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat "/common/home/sl2148/.cache/huggingface/token")
fi

mkdir -p "$LOGDIR"

declare -A EXPECTED_LINES
EXPECTED_LINES[gsm8k]=1319
EXPECTED_LINES[math500]=500
EXPECTED_LINES[aime]=30
EXPECTED_LINES[amc]=40
EXPECTED_LINES[olympiad]=675

# ── helpers ──────────────────────────────────────────────────────────────────

check_gpu_free() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '
}

wait_for_n_gpus() {
    local needed=$1; shift; local gpu_list=("$@")
    while true; do
        local ready_gpus=() status_line=""
        for gpu in "${gpu_list[@]}"; do
            local free_mib=$(check_gpu_free "$gpu")
            status_line="$status_line GPU$gpu:${free_mib}M"
            if [ "$free_mib" -ge "$FREE_THRESHOLD" ] 2>/dev/null; then
                ready_gpus+=("$gpu")
            fi
        done
        if [ "${#ready_gpus[@]}" -ge "$needed" ]; then
            local selected="${ready_gpus[*]:0:$needed}"
            echo "[$(date '+%H:%M:%S')] GPUs ready: $selected |$status_line"
            SELECTED_GPUS=$(echo "$selected" | tr ' ' ',')
            return 0
        fi
        echo "[$(date '+%H:%M:%S')] ${#ready_gpus[@]}/$needed |$status_line"
        sleep "$POLL_INTERVAL"
    done
}

is_complete() {
    local outfile=$1 dataset=$2 expected=${EXPECTED_LINES[$dataset]}
    [ -f "$outfile" ] && [ "$(wc -l < "$outfile")" -ge "$expected" ]
}

get_output_path() {
    local model_san=$(echo "$1" | tr '/' '_')
    if [ "$2" = "gsm8k" ]; then
        echo "$GSM8K_DATA/$model_san/gsm8k_samples.jsonl"
    else
        echo "$MULTI_DATA/$2/$model_san/samples.jsonl"
    fi
}

run_single() {
    local gpu=$1 model=$2 dataset=$3 tp=${4:-1} gpu_mem=${5:-0.90} max_len=${6:-4096} extra=${7:-}
    local outfile=$(get_output_path "$model" "$dataset")
    local model_short=$(echo "$model" | sed 's/.*\///' | sed 's/-Instruct//')
    local logfile="$LOGDIR/${dataset}_${model_short}.log"

    if is_complete "$outfile" "$dataset"; then
        echo "  SKIP $model_short × $dataset (already complete)"
        return 0
    fi

    echo "  RUN  $model_short × $dataset  GPU=$gpu TP=$tp  $(date '+%H:%M:%S')"
    local exit_code=0
    if [ "$dataset" = "gsm8k" ]; then
        CUDA_VISIBLE_DEVICES=$gpu $PYTHON "$GSM8K_SCRIPT" \
            --model "$model" \
            --tensor_parallel_size "$tp" \
            --num_samples 8 --temperature 0.7 --max_tokens 2048 \
            --gpu_memory_utilization "$gpu_mem" --max_model_len "$max_len" \
            $extra \
            > "$logfile" 2>&1 || exit_code=$?
    else
        CUDA_VISIBLE_DEVICES=$gpu $PYTHON "$MULTI_SCRIPT" \
            --model "$model" \
            --dataset "$dataset" \
            --num_samples 16 --temperature 0.7 --max_tokens 2048 \
            --tensor_parallel_size "$tp" --max_model_len "$max_len" \
            --gpu_memory_utilization "$gpu_mem" \
            $extra \
            > "$logfile" 2>&1 || exit_code=$?
    fi

    if [ $exit_code -eq 0 ]; then
        echo "  DONE $model_short × $dataset  $(date '+%H:%M:%S')"
    else
        echo "  FAIL $model_short × $dataset (exit=$exit_code) — see $logfile"
    fi
    return $exit_code
}

run_model_datasets() {
    local gpu=$1 model=$2 tp=$3 gpu_mem=$4 max_len=$5 extra=$6
    shift 6; local datasets=("$@")
    local model_short=$(echo "$model" | sed 's/.*\///' | sed 's/-Instruct//')
    echo "[$(date '+%H:%M:%S')] [$model_short] ${#datasets[@]} datasets on GPU $gpu (TP=$tp)"
    for ds in "${datasets[@]}"; do
        run_single "$gpu" "$model" "$ds" "$tp" "$gpu_mem" "$max_len" "$extra" || true
    done
    echo "[$(date '+%H:%M:%S')] [$model_short] finished all datasets"
}

echo "==========================================="
echo "  Comprehensive Figure 1 Queue"
echo "  Start: $(date)"
echo "==========================================="

###########################################################
# PART A: QWEN experiments (models already cached)
###########################################################

echo ""
echo "╔═══════════════════════════════════════╗"
echo "║  PART A: Qwen remaining experiments   ║"
echo "╚═══════════════════════════════════════╝"

# Phase A1: Qwen-14B on 2 GPUs (aime, amc, olympiad)
echo ""
echo "===== A1: Qwen-14B remainder (TP=2) ====="
wait_for_n_gpus 2 4 5 6 7
run_model_datasets "$SELECTED_GPUS" Qwen/Qwen2.5-14B-Instruct 2 0.90 4096 "" aime amc olympiad
echo "===== A1 complete $(date) ====="

# Phase A2: Qwen-32B on 2 GPUs (math500, aime, amc, olympiad)
echo ""
echo "===== A2: Qwen-32B (TP=2) ====="
wait_for_n_gpus 2 4 5 6 7
run_model_datasets "$SELECTED_GPUS" Qwen/Qwen2.5-32B-Instruct 2 0.92 4096 "" math500 aime amc olympiad
echo "===== A2 complete $(date) ====="

# Phase A3: Qwen-72B on 4 GPUs (math500, aime, amc, olympiad)
echo ""
echo "===== A3: Qwen-72B (TP=4) ====="
wait_for_n_gpus 4 4 5 6 7
run_model_datasets "$SELECTED_GPUS" Qwen/Qwen2.5-72B-Instruct 4 0.95 4096 "" math500 aime amc olympiad
echo "===== A3 complete $(date) ====="

echo ""
echo "╔═══════════════════════════════════════╗"
echo "║  PART A DONE — all Qwen complete      ║"
echo "╚═══════════════════════════════════════╝"

###########################################################
# PART B: LLAMA experiments (needs HF auth for gated repos)
###########################################################

echo ""
echo "╔═══════════════════════════════════════╗"
echo "║  PART B: Llama experiments            ║"
echo "╚═══════════════════════════════════════╝"

# Quick auth check: try to access a small Llama model
echo "Checking Llama model access..."
if ! $PYTHON -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Llama-3.2-1B-Instruct', allow_patterns=['config.json'])
" > /dev/null 2>&1; then
    echo ""
    echo "!!! ERROR: Cannot access Llama models (gated repo)."
    echo "!!! Run:  /common/home/sl2148/anaconda3/envs/fact_yang/bin/huggingface-cli login"
    echo "!!! Then rerun this script to continue with Llama experiments."
    echo ""
    echo "PART A (Qwen) completed successfully. Exiting."
    exit 0
fi

echo "Llama access OK. Downloading models..."
for m in meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct \
         meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.1-70B-Instruct; do
    echo "  Downloading $m ..."
    $PYTHON -c "from huggingface_hub import snapshot_download; snapshot_download('$m')" 2>&1 | tail -1 || echo "  (download may fail for $m, will retry at runtime)"
done

ALL5=(gsm8k math500 aime amc olympiad)

# Phase B1: Small Llama in parallel
echo ""
echo "===== B1: Small Llama models (parallel) ====="
wait_for_n_gpus 3 4 5 6 7

(run_model_datasets 4 meta-llama/Llama-3.2-1B-Instruct 1 0.90 4096 "" "${ALL5[@]}"
 run_model_datasets 4 meta-llama/Llama-3.2-3B-Instruct 1 0.90 4096 "" "${ALL5[@]}") &
PID_A=$!

(run_model_datasets 5 meta-llama/Llama-3.1-8B-Instruct 1 0.90 4096 "" "${ALL5[@]}") &
PID_B=$!

echo "B1 PIDs: $PID_A $PID_B"
wait $PID_A $PID_B || true
echo "===== B1 complete $(date) ====="

# Phase B2: Llama-70B on 4 GPUs
echo ""
echo "===== B2: Llama-70B (TP=4) ====="
wait_for_n_gpus 4 4 5 6 7
run_model_datasets "$SELECTED_GPUS" meta-llama/Llama-3.1-70B-Instruct 4 0.95 4096 "" "${ALL5[@]}"
echo "===== B2 complete $(date) ====="

echo ""
echo "==========================================="
echo "  ALL TASKS DONE at $(date)"
echo "==========================================="
