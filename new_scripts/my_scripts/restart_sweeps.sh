#!/bin/bash
set -e

PYTHON=/common/home/sl2148/anaconda3/envs/fact_yang/bin/python
BASE=/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement
SCRIPT=$BASE/new_scripts/my_scripts/e8_sweep_eval.py
LOGDIR=$BASE/new_scripts/my_scripts/logs

echo "$(date) - Killing any orphan lm_eval processes..."
pkill -f "lm_eval.*steer_hf" 2>/dev/null || true
pkill -f "e8_sweep_eval" 2>/dev/null || true
sleep 3

echo "$(date) - GPU status:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

echo "$(date) - Starting GPT_REWRITE sweep on GPUs 0,1..."
cd $BASE
nohup $PYTHON -u $SCRIPT \
  --model Qwen/Qwen2.5-3B-Instruct \
  --mode GPT_REWRITE \
  --sizes 1 5 10 25 50 \
  --gpus 0 1 \
  --limit 100 \
  --batch_size 16 \
  > $LOGDIR/e8_sweep_gpt_rewrite.log 2>&1 &
GPT_PID=$!
echo "GPT_REWRITE PID: $GPT_PID"

echo "$(date) - Starting LARGE_MODEL sweep on GPUs 2,3,4,5,6,7..."
nohup $PYTHON -u $SCRIPT \
  --model Qwen/Qwen2.5-3B-Instruct \
  --mode LARGE_MODEL \
  --rewrite_model Qwen/Qwen2.5-7B-Instruct \
  --sizes 1 5 10 25 50 \
  --gpus 2 3 4 5 6 7 \
  --limit 100 \
  --batch_size 16 \
  > $LOGDIR/e8_sweep_large_model.log 2>&1 &
LARGE_PID=$!
echo "LARGE_MODEL PID: $LARGE_PID"

echo "$(date) - Waiting 30s for processes to start..."
sleep 30

echo "$(date) - Process check:"
ps aux | grep e8_sweep | grep -v grep || echo "WARNING: No e8_sweep processes found!"

echo "$(date) - GPU status:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

echo "$(date) - Done. PIDs: GPT=$GPT_PID LARGE=$LARGE_PID"
