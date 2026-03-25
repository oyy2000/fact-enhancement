#!/bin/bash
cd /common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement
/common/home/sl2148/anaconda3/envs/fact_yang/bin/python new_scripts/figure1_improved_plot.py \
    --data_dir /common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/new_exps/figure1_sampling_data \
    > /common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/new_scripts/plot_output.log 2>&1
echo "EXIT_CODE=$?" >> /common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/new_scripts/plot_output.log
