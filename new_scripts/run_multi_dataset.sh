#!/bin/bash
# Run Figure 1 validation across multiple datasets for 0.5B-14B models.
# Uses all 8 GPUs: small models get 1 GPU each, 14B gets 2 GPUs.
# Strategy: run one dataset at a time, parallelize across models.
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

NUM_SAMPLES=16   # more samples for small datasets like AIME(30)/AMC(40)

run_model() {
    local gpu=$1
    local model=$2
    local dataset=$3
    local tp=${4:-1}
    local short=$(echo $model | sed 's/.*\///' | sed 's/-Instruct//')
    local logfile="$LOGDIR/${dataset}_${short}.log"

    echo "[$(date +%H:%M:%S)] Starting $short on $dataset (GPU $gpu, TP=$tp)"
    CUDA_VISIBLE_DEVICES=$gpu $PY $SCRIPT \
        --model $model \
        --dataset $dataset \
        --num_samples $NUM_SAMPLES \
        --temperature 0.7 \
        --max_tokens 2048 \
        --tensor_parallel_size $tp \
        --max_model_len 4096 \
        > "$logfile" 2>&1
    echo "[$(date +%H:%M:%S)] Done: $short on $dataset (exit=$?)"
}

for DATASET in math500 aime amc olympiad; do
    echo ""
    echo "=========================================="
    echo "  Dataset: $DATASET"
    echo "=========================================="

    # Launch all models in parallel
    # GPU 0: 0.5B, GPU 1: 1.5B, GPU 2: 3B, GPU 3: 7B, GPU 4-5: 14B
    run_model 0 Qwen/Qwen2.5-0.5B-Instruct $DATASET 1 &
    run_model 1 Qwen/Qwen2.5-1.5B-Instruct $DATASET 1 &
    run_model 2 Qwen/Qwen2.5-3B-Instruct   $DATASET 1 &
    run_model 3 Qwen/Qwen2.5-7B-Instruct   $DATASET 1 &
    run_model 4,5 Qwen/Qwen2.5-14B-Instruct $DATASET 2 &

    # Wait for all models to finish on this dataset
    wait
    echo "[$(date +%H:%M:%S)] All models done for $DATASET"
done

echo ""
echo "All datasets complete!"
