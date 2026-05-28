#!/bin/bash
# Launch recompute_missing_ppl.py — only the 4 chunks missing PPL, on GPUs 0-3
cd /common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement

echo "=== Killing any previous runs ==="
pkill -f "recompute_missing_ppl" 2>/dev/null
sleep 2

echo "=== GPU status ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free --format=csv,noheader,nounits

echo ""
echo "=== Dry run ==="
/common/users/sl2148/anaconda3/envs/fact_yang/bin/python3 -u new_scripts/recompute_missing_ppl.py --dry-run 2>&1

echo ""
echo "=== Launching on GPUs 0,1,2,3 ==="
nohup /common/users/sl2148/anaconda3/envs/fact_yang/bin/python3 -u new_scripts/recompute_missing_ppl.py --gpus 0,1,2,3 > documents/recompute_ppl_run.log 2>&1 &
echo "PID=$!"

echo ""
echo "Monitor with:"
echo "  tail -f documents/recompute_ppl_run.log"
