#!/bin/bash
# Wrapper script for run_pending.py with logging
cd /common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement
exec /common/home/sl2148/anaconda3/envs/fact_yang/bin/python -u new_scripts/run_pending.py \
    --retry-failed --gpus 0,1,2,3 --min-free 30000 \
    >> new_exps/figure1_multi_dataset/logs/tmux_e4_final.log 2>&1
