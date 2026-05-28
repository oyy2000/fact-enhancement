#!/bin/bash
PYTHON=/common/home/sl2148/anaconda3/envs/fact_yang/bin/python
BASE=/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement
SCRIPT=$BASE/new_scripts/my_scripts/e8_sweep_eval.py
LOGDIR=$BASE/new_scripts/my_scripts/logs

echo "$(date) - Killing LARGE_MODEL sweep and its lm_eval children..."
pkill -f "e8_sweep.*LARGE_MODEL" 2>/dev/null || true
sleep 2
pkill -f "lm_eval.*LARGE_MODEL" 2>/dev/null || true
sleep 5

echo "$(date) - GPU status:"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader

echo "$(date) - Finding free GPUs (>30GB free)..."
FREE_GPUS=$($PYTHON -c "
import subprocess
out = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'], text=True)
free = []
for line in out.strip().split('\n'):
    idx, mem = line.split(',')
    if int(mem.strip()) > 30000:
        free.append(idx.strip())
if free:
    print(','.join(free))
else:
    print('2,5')
")
echo "Free GPUs: $FREE_GPUS"

# Convert comma-separated to space-separated for argparse
GPUS_ARGS=$(echo $FREE_GPUS | tr ',' ' ')

echo "$(date) - Starting LARGE_MODEL sweep on GPUs: $GPUS_ARGS"
cd $BASE
nohup $PYTHON -u $SCRIPT \
  --model Qwen/Qwen2.5-3B-Instruct \
  --mode LARGE_MODEL \
  --rewrite_model Qwen/Qwen2.5-7B-Instruct \
  --sizes 1 5 10 25 50 \
  --gpus $GPUS_ARGS \
  --limit 100 \
  --batch_size 16 \
  > $LOGDIR/e8_sweep_large_model.log 2>&1 &
echo "LARGE_MODEL PID: $!"

sleep 30
echo "$(date) - Verification:"
ps aux | grep "e8_sweep.*LARGE" | grep -v grep | head -3
echo "$(date) - GPU status:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "$(date) - Done."
