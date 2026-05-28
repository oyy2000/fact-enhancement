#!/bin/bash
set -uo pipefail

ROOT="/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement"
PY="/common/users/sl2148/anaconda3/envs/fact_yang/bin/python"
cd "$ROOT"

VEC="$ROOT/exps/logiqa_densesteer/vectors/Qwen_Qwen2.5-3B-Instruct/N50_infamily_7b/Qwen_Qwen2.5-3B-Instruct_applied/steering_vector.pt"
SWEEP_DIR="$ROOT/exps/logiqa_densesteer/sweep"
GPU="${1:-0}"

HF_TOKEN_PATH="$ROOT/new_exps/.cache/huggingface/token"
if [ -f "$HF_TOKEN_PATH" ]; then
    export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

run_one() {
    local LAYER="$1" LAM="$2" TAG="$3"
    local OUTDIR="$SWEEP_DIR/$TAG/L${LAYER}/lam_${LAM}"

    if find "$OUTDIR" -name "results_*.json" 2>/dev/null | head -1 | grep -q .; then
        echo "[CACHED] $TAG L$LAYER λ=$LAM"
        return 0
    fi
    mkdir -p "$OUTDIR"
    echo "[RUN] $TAG L$LAYER λ=$LAM"
    CUDA_VISIBLE_DEVICES=$GPU $PY -m lm_eval \
        --model steer_hf \
        --model_args "pretrained=Qwen/Qwen2.5-3B-Instruct,dtype=float16,steer_layer=$LAYER,steer_lambda=$LAM,steer_vec_path=$VEC" \
        --tasks logiqa \
        --device cuda:0 \
        --num_fewshot 0 \
        --batch_size 16 \
        --output_path "$OUTDIR" \
        --log_samples \
        --trust_remote_code \
        --apply_chat_template 2>&1 | tail -3
}

# Baseline
BASELINE_DIR="$SWEEP_DIR/BASELINE/L0/lam_0"
if ! find "$BASELINE_DIR" -name "results_*.json" 2>/dev/null | head -1 | grep -q .; then
    echo "[RUN] BASELINE"
    run_one 0 0.0 BASELINE
fi

echo "=== InFamily sweep ==="
for LAYER in 2 4 6 8 10 12 14 16 18; do
    for LAM in -1.0 -0.5 -0.3 0.3 0.45 0.5 0.7 1.0; do
        run_one $LAYER $LAM InFamily || true
    done
done

echo "=== InFamily sweep DONE ==="
