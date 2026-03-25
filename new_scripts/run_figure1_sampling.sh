#!/bin/bash
# Run sampling-based GSM8K evaluation for all Qwen models in parallel.
# Each model gets its own GPU(s). Generates 8 responses per question.

PYTHON=/common/home/sl2148/anaconda3/envs/fact_yang/bin/python
BASE=/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement
LOGDIR=$BASE/new_exps/figure1_sampling_data
cd $BASE

mkdir -p $LOGDIR

NUM_SAMPLES=8
TEMP=0.7

echo "=== Figure 1 Sampling Evaluation ==="
echo "Start time: $(date)"
echo "Generating ${NUM_SAMPLES} samples/question at temperature=${TEMP}"
echo "Output dir: $LOGDIR"
echo ""

# 0.5B on GPU 0 (model ~1GB, very fast) - batch_size=8
CUDA_VISIBLE_DEVICES=0 $PYTHON $BASE/new_scripts/figure1_sampling_eval.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --num_samples $NUM_SAMPLES --temperature $TEMP \
    --batch_size 8 \
    2>&1 | tee $LOGDIR/log_0.5B.txt &

# 1.5B on GPU 1 (model ~3GB) - batch_size=4
CUDA_VISIBLE_DEVICES=1 $PYTHON $BASE/new_scripts/figure1_sampling_eval.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --num_samples $NUM_SAMPLES --temperature $TEMP \
    --batch_size 4 \
    2>&1 | tee $LOGDIR/log_1.5B.txt &

# 3B on GPU 2 (model ~6GB) - batch_size=4
CUDA_VISIBLE_DEVICES=2 $PYTHON $BASE/new_scripts/figure1_sampling_eval.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --num_samples $NUM_SAMPLES --temperature $TEMP \
    --batch_size 4 \
    2>&1 | tee $LOGDIR/log_3B.txt &

# 7B on GPU 3 (model ~14GB) - batch_size=2
CUDA_VISIBLE_DEVICES=3 $PYTHON $BASE/new_scripts/figure1_sampling_eval.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num_samples $NUM_SAMPLES --temperature $TEMP \
    --batch_size 2 \
    2>&1 | tee $LOGDIR/log_7B.txt &

# 14B on GPUs 4,5 (model ~28GB fp16, needs 2 GPUs) - batch_size=1
CUDA_VISIBLE_DEVICES=4,5 $PYTHON $BASE/new_scripts/figure1_sampling_eval.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --num_samples $NUM_SAMPLES --temperature $TEMP \
    --batch_size 1 \
    2>&1 | tee $LOGDIR/log_14B.txt &

echo "All 5 models launched. PIDs: $(jobs -p)"
echo "Logs in $LOGDIR/log_*.txt"
wait
echo ""
echo "=== All done at $(date) ==="
