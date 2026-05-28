#!/bin/bash
# Overnight E12 runner: BBH + HotpotQA for all 3 modes
# BBH on GPUs 0,1,2; HotpotQA on GPUs 3,4,5
# Each task gets its own dedicated GPU to avoid OOM

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
    local SAFE_MODEL=$(echo $MODEL | tr '/' '__')
    local LAM_TAG=$(python3 -c "print(f'{$LAM:.6f}'.rstrip('0').rstrip('.'))")
    local OUTDIR="$OUTBASE/$MODE/$SAFE_MODEL/$TASK/L${LAYER}/lam_${LAM_TAG}"
    local LOGFILE="$LOGDIR/${MODE}_${TASK}_L${LAYER}_lam${LAM_TAG}.log"

    # Skip if results already exist
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
        echo "[FAIL] $MODE $TASK L$LAYER lam=$LAM (rc=$RC) $(date)"
    fi
    return $RC
}

echo "============================================"
echo " Overnight E12 Experiment Runner"
echo " Started: $(date)"
echo " GPUs: 0-7 all available"
echo "============================================"

# ====== BBH: 3 modes in parallel (GPUs 0,1,2) ======
echo ""
echo ">>> Launching BBH CoT Zero-shot (3 modes)..."
run_task 0 bbh_cot_zeroshot BASELINE    0 0.0  "$GPT_VEC" 16 1024 &
PID_BBH_BASE=$!
run_task 1 bbh_cot_zeroshot GPT_REWRITE 6 4.0  "$GPT_VEC" 16 1024 &
PID_BBH_GPT=$!
run_task 2 bbh_cot_zeroshot LARGE_MODEL 6 0.45 "$LM_VEC"  16 1024 &
PID_BBH_LM=$!

# ====== HotpotQA: 3 modes in parallel (GPUs 3,4,5) - batch=1 for long context ======
echo ""
echo ">>> Launching HotpotQA (3 modes, batch=1 for long context)..."
run_task 3 longbench_hotpotqa BASELINE    0 0.0  "$GPT_VEC" 1 512 &
PID_HP_BASE=$!
run_task 4 longbench_hotpotqa GPT_REWRITE 6 4.0  "$GPT_VEC" 1 512 &
PID_HP_GPT=$!
run_task 5 longbench_hotpotqa LARGE_MODEL 6 0.45 "$LM_VEC"  1 512 &
PID_HP_LM=$!

echo ""
echo ">>> All 6 tasks launched. PIDs:"
echo "    BBH:      BASELINE=$PID_BBH_BASE GPT=$PID_BBH_GPT LM=$PID_BBH_LM"
echo "    HotpotQA: BASELINE=$PID_HP_BASE  GPT=$PID_HP_GPT  LM=$PID_HP_LM"
echo ""

# Wait for all
wait $PID_BBH_BASE $PID_BBH_GPT $PID_BBH_LM $PID_HP_BASE $PID_HP_GPT $PID_HP_LM

echo ""
echo "============================================"
echo " All tasks finished: $(date)"
echo "============================================"

# Summary
echo ""
echo ">>> Results found:"
find "$OUTBASE" -name "results_*.json" | while read f; do echo "  $f"; done

echo ""
echo ">>> Completed tasks:"
for MODE in BASELINE GPT_REWRITE LARGE_MODEL; do
    for TASK in mmlu gpqa_main_zeroshot bbh_cot_zeroshot longbench_hotpotqa; do
        SAFE_MODEL=$(echo $MODEL | tr '/' '__')
        COUNT=$(find "$OUTBASE/$MODE/$SAFE_MODEL/$TASK" -name "results_*.json" 2>/dev/null | wc -l)
        if [ "$COUNT" -gt 0 ]; then
            echo "  [OK] $MODE / $TASK"
        else
            echo "  [--] $MODE / $TASK"
        fi
    done
done
