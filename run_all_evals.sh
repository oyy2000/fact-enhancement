#!/bin/bash
# Run all pending evaluations in parallel on different GPUs
PYTHON=/common/home/sl2148/anaconda3/envs/fact_yang/bin/python
BASE=/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement
cd $BASE

MODEL="Qwen/Qwen2.5-3B-Instruct"
LAYER=6
LAM=4.0

eval_gsm8k() {
    local GPU=$1
    local VEC_PATH=$2
    local OUT_DIR=$3
    local TAG=$4
    
    echo "[GPU $GPU] Starting eval: $TAG"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m lm_eval \
        --model steer_hf \
        --model_args "pretrained=$MODEL,dtype=float16,steer_layer=$LAYER,steer_lambda=$LAM,steer_vec_path=$VEC_PATH,trust_remote_code=True" \
        --tasks gsm8k_cot_zeroshot_unified \
        --batch_size 64 \
        --num_fewshot 0 \
        --output_path "$OUT_DIR" \
        --log_samples \
        --trust_remote_code \
        --gen_kwargs "do_sample=False,temperature=0,max_gen_toks=2048" \
        2>&1 | tail -5
    echo "[GPU $GPU] Done: $TAG (exit=$?)"
}

# Control experiment: random compression
eval_gsm8k 0 \
    "$BASE/control_experiments/Qwen_Qwen2.5-3B-Instruct/vectors_random_merge/steering_vector.pt" \
    "$BASE/control_experiments/Qwen_Qwen2.5-3B-Instruct/eval_random_merge_L6_lam4.0" \
    "control_random_merge" &

# Calibration ablation: N=1
eval_gsm8k 1 \
    "$BASE/calibration_ablation/Qwen_Qwen2.5-3B-Instruct/GPT_REWRITE/vectors_N1/steering_vector.pt" \
    "$BASE/calibration_ablation/Qwen_Qwen2.5-3B-Instruct/GPT_REWRITE/eval_N1_L6_lam4.0" \
    "calibration_N1" &

# Calibration ablation: N=5
eval_gsm8k 2 \
    "$BASE/calibration_ablation/Qwen_Qwen2.5-3B-Instruct/GPT_REWRITE/vectors_N5/steering_vector.pt" \
    "$BASE/calibration_ablation/Qwen_Qwen2.5-3B-Instruct/GPT_REWRITE/eval_N5_L6_lam4.0" \
    "calibration_N5" &

# Calibration ablation: N=10
eval_gsm8k 3 \
    "$BASE/calibration_ablation/Qwen_Qwen2.5-3B-Instruct/GPT_REWRITE/vectors_N10/steering_vector.pt" \
    "$BASE/calibration_ablation/Qwen_Qwen2.5-3B-Instruct/GPT_REWRITE/eval_N10_L6_lam4.0" \
    "calibration_N10" &

# Calibration ablation: N=25
eval_gsm8k 4 \
    "$BASE/calibration_ablation/Qwen_Qwen2.5-3B-Instruct/GPT_REWRITE/vectors_N25/steering_vector.pt" \
    "$BASE/calibration_ablation/Qwen_Qwen2.5-3B-Instruct/GPT_REWRITE/eval_N25_L6_lam4.0" \
    "calibration_N25" &

# Calibration ablation: N=50
eval_gsm8k 5 \
    "$BASE/calibration_ablation/Qwen_Qwen2.5-3B-Instruct/GPT_REWRITE/vectors_N50/steering_vector.pt" \
    "$BASE/calibration_ablation/Qwen_Qwen2.5-3B-Instruct/GPT_REWRITE/eval_N50_L6_lam4.0" \
    "calibration_N50" &

echo "All 6 evaluations launched. Waiting..."
wait
echo "All evaluations completed."
