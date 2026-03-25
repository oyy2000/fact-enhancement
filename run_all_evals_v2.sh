#!/bin/bash
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
    
    echo "[GPU $GPU] Starting: $TAG"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m lm_eval \
        --model steer_hf \
        --model_args "pretrained=$MODEL,dtype=float16,steer_layer=$LAYER,steer_lambda=$LAM,steer_vec_path=$VEC_PATH,trust_remote_code=True" \
        --tasks gsm8k_cot_zeroshot_unified \
        --batch_size 4 \
        --num_fewshot 0 \
        --output_path "$OUT_DIR" \
        --log_samples \
        --trust_remote_code \
        --gen_kwargs "do_sample=False,temperature=0,max_gen_toks=2048" \
        2>&1 | tail -3
    echo "[GPU $GPU] Done: $TAG"
}

# Calibration ablations on GPUs 1-5
eval_gsm8k 1 "$BASE/calibration_ablation/Qwen_Qwen2.5-3B-Instruct/GPT_REWRITE/vectors_N1/steering_vector.pt" \
    "$BASE/calibration_ablation/Qwen_Qwen2.5-3B-Instruct/GPT_REWRITE/eval_N1_L6_lam4.0" "N1" &
eval_gsm8k 2 "$BASE/calibration_ablation/Qwen_Qwen2.5-3B-Instruct/GPT_REWRITE/vectors_N5/steering_vector.pt" \
    "$BASE/calibration_ablation/Qwen_Qwen2.5-3B-Instruct/GPT_REWRITE/eval_N5_L6_lam4.0" "N5" &
eval_gsm8k 3 "$BASE/calibration_ablation/Qwen_Qwen2.5-3B-Instruct/GPT_REWRITE/vectors_N10/steering_vector.pt" \
    "$BASE/calibration_ablation/Qwen_Qwen2.5-3B-Instruct/GPT_REWRITE/eval_N10_L6_lam4.0" "N10" &
eval_gsm8k 4 "$BASE/calibration_ablation/Qwen_Qwen2.5-3B-Instruct/GPT_REWRITE/vectors_N25/steering_vector.pt" \
    "$BASE/calibration_ablation/Qwen_Qwen2.5-3B-Instruct/GPT_REWRITE/eval_N25_L6_lam4.0" "N25" &
eval_gsm8k 5 "$BASE/calibration_ablation/Qwen_Qwen2.5-3B-Instruct/GPT_REWRITE/vectors_N50/steering_vector.pt" \
    "$BASE/calibration_ablation/Qwen_Qwen2.5-3B-Instruct/GPT_REWRITE/eval_N50_L6_lam4.0" "N50" &

echo "5 calibration evals launched on GPUs 1-5. Waiting..."
wait
echo "All calibration evals done."
