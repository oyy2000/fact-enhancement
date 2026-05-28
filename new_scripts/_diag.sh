#!/bin/bash
echo "=== LOG ==="
cat /common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/new_exps/figure1_multi_dataset/logs/tmux_e4_final.log 2>&1
echo "=== TMUX ==="
tmux ls 2>&1
echo "=== PS ==="
ps aux | grep run_pending | grep -v grep 2>&1
echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>&1
echo "=== DONE ==="
