#!/bin/bash
PYTHON=/common/home/sl2148/anaconda3/envs/fact_yang/bin/python
BASE=/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement
cd $BASE

MODEL="Qwen/Qwen2.5-3B-Instruct"
LAYER=6
LAM=4.0
BS=8

eval_gsm8k() {
    local GPU=$1
    local VEC_PATH=$2
    local OUT_DIR=$3
    local TAG=$4
    
    echo "[GPU $GPU] Starting: $TAG at $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m lm_eval \
        --model steer_hf \
        --model_args "pretrained=$MODEL,dtype=float16,steer_layer=$LAYER,steer_lambda=$LAM,steer_vec_path=$VEC_PATH,trust_remote_code=True" \
        --tasks gsm8k_cot_zeroshot_unified \
        --batch_size $BS \
        --num_fewshot 0 \
        --output_path "$OUT_DIR" \
        --log_samples \
        --trust_remote_code \
        --gen_kwargs "do_sample=False,temperature=0,max_gen_toks=2048" 2>&1
    echo "[GPU $GPU] Done: $TAG at $(date +%H:%M:%S) exit=$?"
}

# GPU 0: Control experiment
eval_gsm8k 0 "$BASE/control_experiments/Qwen_Qwen2.5-3B-Instruct/vectors_random_merge/steering_vector.pt" \
    "$BASE/control_experiments/Qwen_Qwen2.5-3B-Instruct/eval_random_merge_L6_lam4.0" "control" &

# GPUs 1-5: Calibration ablation N=1,5,10,25,50
for N in 1 5 10 25 50; do
    case $N in
        1)  GPU=1;;
        5)  GPU=2;;
        10) GPU=3;;
        25) GPU=4;;
        50) GPU=5;;
    esac
    eval_gsm8k $GPU "$BASE/calibration_ablation/Qwen_Qwen2.5-3B-Instruct/GPT_REWRITE/vectors_N${N}/steering_vector.pt" \
        "$BASE/calibration_ablation/Qwen_Qwen2.5-3B-Instruct/GPT_REWRITE/eval_N${N}_L6_lam4.0" "N${N}" &
done

echo "All 6 evals launched on GPUs 0-5. Waiting..."
wait
echo "=== ALL DONE at $(date +%H:%M:%S) ==="
