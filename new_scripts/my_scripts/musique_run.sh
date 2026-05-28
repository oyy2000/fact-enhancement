#!/bin/bash
# MuSiQue experiment runner - uses GPUs 3,4,5 (freed from HotpotQA)

PYTHON=/common/users/sl2148/anaconda3/envs/fact_yang/bin/python
BASE=/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement
HF_TOKEN=$(cat $BASE/new_exps/.cache/huggingface/token 2>/dev/null)
export HF_TOKEN HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

MODEL="Qwen/Qwen2.5-3B-Instruct"
GPT_VEC="$BASE/exps/gpt_rewrites_unified_new/Qwen_Qwen2.5-3B-Instruct/vectors_50_old/Qwen_Qwen2.5-3B-Instruct_applied/steering_vector.pt"
LM_VEC="$BASE/exps/large_model_rewrites_unified_new/Qwen_Qwen2.5-3B-Instruct/vectors_50_paired_Qwen_Qwen2.5-7B-Instruct/Qwen_Qwen2.5-3B-Instruct_applied/steering_vector.pt"

OUTBASE="$BASE/exps/steer_runs_e11_e12"
LOGDIR="$OUTBASE/overnight_logs"
mkdir -p "$LOGDIR"

run_task() {
    local GPU=$1 TASK=$2 MODE=$3 LAYER=$4 LAM=$5 VEC=$6 BATCH=$7 MAXGEN=$8
    local SAFE_MODEL=$(echo $MODEL | tr '/' '_')
    local LAM_TAG=$(python3 -c "print(f'{$LAM:.6f}'.rstrip('0').rstrip('.'))")
    local OUTDIR="$OUTBASE/$MODE/$SAFE_MODEL/$TASK/L${LAYER}/lam_${LAM_TAG}"
    local LOGFILE="$LOGDIR/${MODE}_${TASK}_L${LAYER}_lam${LAM_TAG}.log"

    if find "$OUTDIR" -name "results_*.json" 2>/dev/null | grep -q .; then
        echo "[SKIP] $MODE $TASK L$LAYER lam=$LAM - already done"
        return 0
    fi

    mkdir -p "$OUTDIR"
    echo "[START] GPU=$GPU $MODE $TASK L$LAYER lam=$LAM batch=$BATCH $(date)"

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m lm_eval \
        --model steer_hf \
        --model_args "pretrained=$MODEL,dtype=float16,steer_layer=$LAYER,steer_lambda=$LAM,steer_vec_path=$VEC" \
        --tasks "$TASK" \
        --device cuda:0 \
        --num_fewshot 0 \
        --batch_size "$BATCH" \
        --gen_kwargs "do_sample=false,temperature=0,max_gen_toks=$MAXGEN" \
        --output_path "$OUTDIR" \
        --log_samples \
        --trust_remote_code \
        --apply_chat_template \
        > "$LOGFILE" 2>&1

    local RC=$?
    if [ $RC -eq 0 ]; then
        echo "[DONE] $MODE $TASK L$LAYER lam=$LAM (rc=$RC) $(date)"
    else
        echo "[FAIL] $MODE $TASK L$LAYER lam=$LAM (rc=$RC) $(date). Check $LOGFILE"
    fi
    return $RC
}

echo "============================================"
echo " MuSiQue Experiment Runner"
echo " Started: $(date)"
echo " Using GPUs 3,4,5"
echo "============================================"

run_task 3 longbench_musique BASELINE    0 0.0  "$GPT_VEC" 1 512 &
PID1=$!
run_task 4 longbench_musique GPT_REWRITE 6 4.0  "$GPT_VEC" 1 512 &
PID2=$!
run_task 5 longbench_musique LARGE_MODEL 6 0.45 "$LM_VEC"  1 512 &
PID3=$!

echo ">>> MuSiQue PIDs: BASELINE=$PID1 GPT=$PID2 LM=$PID3"

wait $PID1 $PID2 $PID3

echo ""
echo "============================================"
echo " MuSiQue finished: $(date)"
echo "============================================"
echo ">>> MuSiQue results:"
find "$OUTBASE" -path "*/longbench_musique/*/results_*.json" 2>/dev/null | while read f; do echo "  $f"; done
