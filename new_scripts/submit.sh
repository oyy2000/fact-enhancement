#!/bin/bash
# Submit pending tasks. Run repeatedly to gradually complete all experiments.
# Usage:
#   bash new_scripts/submit.sh               # auto-detect free GPUs, run in background
#   bash new_scripts/submit.sh --status      # just check status
#   bash new_scripts/submit.sh --dry-run     # show plan only
#   bash new_scripts/submit.sh --gpus 0,1,2  # use specific GPUs
#   bash new_scripts/submit.sh --retry-failed # retry all failed tasks

cd /common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement

PY=/common/home/sl2148/anaconda3/envs/fact_yang/bin/python
LOGFILE=new_exps/figure1_multi_dataset/logs/scheduler_$(date +%Y%m%d_%H%M%S).log

if [[ "$*" == *"--status"* ]] || [[ "$*" == *"--dry-run"* ]]; then
    $PY new_scripts/run_pending.py "$@"
else
    echo "Submitting tasks in background..."
    echo "Log: $LOGFILE"
    nohup $PY -u new_scripts/run_pending.py "$@" > "$LOGFILE" 2>&1 &
    echo "PID: $!"
    sleep 3
    head -30 "$LOGFILE" 2>/dev/null
fi
