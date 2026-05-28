#!/usr/bin/env bash
set -euo pipefail

cd /common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement
PY=/common/home/sl2148/anaconda3/envs/fact_yang/bin/python
LOG=new_exps/retry_remaining.log
TASKS=new_exps/tasks.json

echo "=== retry_remaining started at $(date) ===" >> "$LOG"

MAX_ROUNDS=60
INTERVAL=600  # 10 minutes

for round in $(seq 1 $MAX_ROUNDS); do
    pending=$($PY -c "
import json
d=json.load(open('$TASKS'))
n=sum(1 for t in d['tasks'] if t['status'] in ('pending','failed') and not t.get('skip',False))
print(n)
")

    if [ "$pending" -eq 0 ]; then
        echo "[$(date)] All non-skipped tasks done!" | tee -a "$LOG"
        break
    fi

    echo "[$(date)] Round $round: $pending tasks remaining" | tee -a "$LOG"

    free_gpus=$($PY -c "
import subprocess, re
out = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits']).decode()
gpus = []
for line in out.strip().split('\n'):
    idx, free = line.split(',')
    if int(free.strip()) >= 15000:
        gpus.append(idx.strip())
print(','.join(gpus) if gpus else '')
")

    if [ -z "$free_gpus" ]; then
        echo "[$(date)] No GPUs with >=15GB free. Waiting ${INTERVAL}s..." | tee -a "$LOG"
        sleep $INTERVAL
        continue
    fi

    echo "[$(date)] Free GPUs: $free_gpus" | tee -a "$LOG"
    $PY -u new_scripts/run_pending.py --retry-failed --gpus "$free_gpus" >> "$LOG" 2>&1 || true

    sleep 30
done

echo "[$(date)] Updating Excel..." | tee -a "$LOG"
$PY -u -c "
import json, glob, os
from pathlib import Path
from datetime import datetime

BASE = Path('new_exps')
results = {}
patterns = [
    'figure1_sampling_results/*/results_*.jsonl',
    'multi_dataset_results/*/results_*.jsonl',
]
for pat in patterns:
    for f in glob.glob(str(BASE / pat)):
        parts = Path(f).parts
        model_dir = parts[-2]
        fname = parts[-1]
        import re
        m_ds = re.match(r'results_(.+)\.jsonl', fname)
        if not m_ds:
            m_ds = re.match(r'results_.*_(.+)\.jsonl', fname)
        if m_ds:
            ds = m_ds.group(1)
        else:
            continue
        lines = open(f).readlines()
        if not lines:
            continue
        last = json.loads(lines[-1])
        key = (model_dir, ds)
        results[key] = last

print(f'Found {len(results)} result files')
for k, v in sorted(results.items()):
    acc = v.get('accuracy', v.get('acc', 'N/A'))
    steps = v.get('avg_steps', 'N/A')
    tokens = v.get('avg_tokens', 'N/A')
    print(f'  {k[0]:40s} x {k[1]:10s} acc={acc}  steps={steps}  tokens={tokens}')
" >> "$LOG" 2>&1 || true

echo "=== retry_remaining finished at $(date) ===" >> "$LOG"
