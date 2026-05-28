#!/bin/bash
set -e
echo "Starting plot generation..."
cd /common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement
MPLBACKEND=Agg /common/users/sl2148/anaconda3/envs/fact_yang/bin/python3 new_scripts/plot_control_ppl.py
echo "Checking output..."
ls -la documents/control_ppl_vs_lambda_L6.png 2>&1 || echo "PNG NOT FOUND"
ls -la documents/control_ppl_grid_eval.json 2>&1 || echo "JSON NOT FOUND"
echo "SCRIPT_DONE"
