#!/bin/bash
set -e
export MPLBACKEND=Agg
cd /common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement
echo "Starting plot generation..."
/common/users/sl2148/anaconda3/envs/fact_yang/bin/python3 -u new_scripts/_plot_joint_metrics.py
echo "Checking output..."
ls -la documents/joint_*.png 2>&1
echo "SCRIPT_DONE"
