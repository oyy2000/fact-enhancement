#!/bin/bash
set -euo pipefail

ROOT="/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement"
PY="/common/users/sl2148/anaconda3/envs/fact_yang/bin/python"
cd "$ROOT"

source .env 2>/dev/null || true
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export BASE_URL="${BASE_URL:-}"
export OPENAI_BASE_URL="${BASE_URL:-}"

EXPS="$ROOT/exps/logiqa_densesteer"
SAMPLE_DIR="$EXPS/samples"
PAIR_DIR="$EXPS/paired"
GPT_DIR="$EXPS/gpt_rewrites/Qwen_Qwen2.5-3B-Instruct"
VEC_DIR="$EXPS/vectors/Qwen_Qwen2.5-3B-Instruct"
SWEEP_DIR="$EXPS/sweep"

mkdir -p "$SAMPLE_DIR" "$PAIR_DIR" "$GPT_DIR" "$VEC_DIR" "$SWEEP_DIR"

JSONL_3B="$SAMPLE_DIR/Qwen_Qwen2.5-3B-Instruct_logiqa_cot_train.jsonl"
JSONL_7B="$SAMPLE_DIR/Qwen_Qwen2.5-7B-Instruct_logiqa_cot_train.jsonl"
PAIR_JSON="$PAIR_DIR/Qwen3B_paired_Qwen7B_logiqa.json"
GPT_JSON="$GPT_DIR/rewritten_old.json"
DENSE_VEC="$VEC_DIR/N50_dense_gpt_old/Qwen_Qwen2.5-3B-Instruct_applied/steering_vector.pt"
INFAM_VEC="$VEC_DIR/N50_infamily_7b/Qwen_Qwen2.5-3B-Instruct_applied/steering_vector.pt"

LOG="$EXPS/pipeline.log"
> "$LOG"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"
}

log "===== STARTING FULL PIPELINE ====="

# ======== Step 0: Generate CoT with vLLM (3B + 7B in parallel) ========
NEED_3B=false
NEED_7B=false
[ ! -f "$JSONL_3B" ] || [ "$(wc -l < "$JSONL_3B")" -lt 100 ] && NEED_3B=true
[ ! -f "$JSONL_7B" ] || [ "$(wc -l < "$JSONL_7B")" -lt 100 ] && NEED_7B=true

if $NEED_3B && $NEED_7B; then
    log "Step 0: Generating 3B (GPU 0) + 7B (GPU 1) in parallel with vLLM..."
    (
        CUDA_VISIBLE_DEVICES=0 $PY -u new_scripts/my_scripts/logiqa_generate_cot_vllm.py \
            --model Qwen/Qwen2.5-3B-Instruct \
            --split train --limit 800 \
            --out_jsonl "$JSONL_3B" 2>&1 | tee -a "$LOG.3b"
    ) &
    PID_3B=$!
    (
        CUDA_VISIBLE_DEVICES=1 $PY -u new_scripts/my_scripts/logiqa_generate_cot_vllm.py \
            --model Qwen/Qwen2.5-7B-Instruct \
            --split train --limit 800 \
            --out_jsonl "$JSONL_7B" 2>&1 | tee -a "$LOG.7b"
    ) &
    PID_7B=$!
    log "  3B PID=$PID_3B, 7B PID=$PID_7B"
    wait $PID_3B
    log "  3B done ($(wc -l < "$JSONL_3B") lines)"
    wait $PID_7B
    log "  7B done ($(wc -l < "$JSONL_7B") lines)"
elif $NEED_3B; then
    log "Step 0a: Generating 3B with vLLM..."
    CUDA_VISIBLE_DEVICES=0 $PY -u new_scripts/my_scripts/logiqa_generate_cot_vllm.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --split train --limit 800 \
        --out_jsonl "$JSONL_3B" 2>&1 | tee -a "$LOG"
    log "  3B done ($(wc -l < "$JSONL_3B") lines)"
elif $NEED_7B; then
    log "Step 0b: Generating 7B with vLLM..."
    CUDA_VISIBLE_DEVICES=1 $PY -u new_scripts/my_scripts/logiqa_generate_cot_vllm.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --split train --limit 800 \
        --out_jsonl "$JSONL_7B" 2>&1 | tee -a "$LOG"
    log "  7B done ($(wc -l < "$JSONL_7B") lines)"
else
    log "Step 0 SKIP: both JSONL files exist"
fi

# ======== Step 1a: GPT rewrite (while pairing can happen in parallel) ========
if [ ! -f "$GPT_JSON" ] || [ "$(stat -c%s "$GPT_JSON" 2>/dev/null || echo 0)" -lt 1000 ]; then
    if [ -z "$OPENAI_API_KEY" ]; then
        log "Step 1a SKIP: No OPENAI_API_KEY"
    else
        log "Step 1a: GPT-5.1 dense rewrite..."
        $PY -u 00_gpt_modification.py \
            --in_jsonl "$JSONL_3B" \
            --out_json "$GPT_JSON" \
            --prompt_style old \
            --rewrite_last_n 100000 2>&1 | tee -a "$LOG"
        log "Step 1a DONE ($(wc -c < "$GPT_JSON") bytes)"
    fi
else
    log "Step 1a SKIP: GPT rewrite exists"
fi

# ======== Step 1b: Pair 3B/7B ========
if [ ! -f "$PAIR_JSON" ] || [ "$(stat -c%s "$PAIR_JSON" 2>/dev/null || echo 0)" -lt 1000 ]; then
    log "Step 1b: Pairing 3B/7B..."
    $PY -u new_scripts/my_scripts/logiqa_pair_infamily.py \
        --small_jsonl "$JSONL_3B" \
        --large_jsonl "$JSONL_7B" \
        --out_json "$PAIR_JSON" 2>&1 | tee -a "$LOG"
    log "Step 1b DONE"
else
    log "Step 1b SKIP: pair JSON exists"
fi

# ======== Step 2: Extract steering vectors ========
if [ ! -f "$DENSE_VEC" ] && [ -f "$GPT_JSON" ]; then
    log "Step 2a: Extracting DenseSteer vector..."
    CUDA_VISIBLE_DEVICES=0 $PY -u new_scripts/my_scripts/logiqa_extract_steering.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --in_path "$GPT_JSON" \
        --num_examples 50 \
        --layers "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18" \
        --tag dense_gpt_old \
        --domain logiqa 2>&1 | tee -a "$LOG"
    log "Step 2a DONE"
elif [ -f "$DENSE_VEC" ]; then
    log "Step 2a SKIP: DenseSteer vector exists"
else
    log "Step 2a SKIP: GPT rewrite file missing"
fi

if [ ! -f "$INFAM_VEC" ] && [ -f "$PAIR_JSON" ]; then
    log "Step 2b: Extracting InFamily vector..."
    CUDA_VISIBLE_DEVICES=0 $PY -u new_scripts/my_scripts/logiqa_extract_steering.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --in_path "$PAIR_JSON" \
        --num_examples 50 \
        --layers "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18" \
        --tag infamily_7b \
        --domain logiqa 2>&1 | tee -a "$LOG"
    log "Step 2b DONE"
elif [ -f "$INFAM_VEC" ]; then
    log "Step 2b SKIP: InFamily vector exists"
else
    log "Step 2b SKIP: pair JSON missing"
fi

# ======== Step 3: Lambda × Layer sweep on logiqa ========
log "Step 3: Lambda × Layer sweep on logiqa"

HF_TOKEN_PATH="$ROOT/new_exps/.cache/huggingface/token"
if [ -f "$HF_TOKEN_PATH" ]; then
    export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

run_sweep() {
    local VEC="$1" LAYER="$2" LAM="$3" TAG="$4" GPU="${5:-0}"
    local OUTDIR="$SWEEP_DIR/$TAG/L${LAYER}/lam_${LAM}"

    if find "$OUTDIR" -name "results_*.json" 2>/dev/null | head -1 | grep -q .; then
        return 0
    fi

    mkdir -p "$OUTDIR"
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
    ANY_VEC=""
    [ -f "$DENSE_VEC" ] && ANY_VEC="$DENSE_VEC"
    [ -z "$ANY_VEC" ] && [ -f "$INFAM_VEC" ] && ANY_VEC="$INFAM_VEC"
    if [ -n "$ANY_VEC" ]; then
        log "  Running baseline..."
        run_sweep "$ANY_VEC" 0 0.0 BASELINE 0 || true
    fi
fi

# Sweep DenseSteer (GPU 0) and InFamily (GPU 1) in parallel
sweep_dense() {
    if [ ! -f "$DENSE_VEC" ]; then return; fi
    for LAYER in 2 4 6 8 10 12 14 16 18; do
        for LAM in -2.0 -1.0 -0.5 0.5 1.0 2.0 3.0 4.0; do
            run_sweep "$DENSE_VEC" $LAYER $LAM DenseSteer 0 || true
        done
    done
}

sweep_infamily() {
    if [ ! -f "$INFAM_VEC" ]; then return; fi
    for LAYER in 2 4 6 8 10 12 14 16 18; do
        for LAM in -1.0 -0.5 -0.3 0.3 0.45 0.5 0.7 1.0; do
            run_sweep "$INFAM_VEC" $LAYER $LAM InFamily 1 || true
        done
    done
}

log "  Starting DenseSteer sweep (GPU 0) + InFamily sweep (GPU 1) in parallel"
sweep_dense &
PID_SD=$!
sweep_infamily &
PID_SI=$!
wait $PID_SD 2>/dev/null || true
log "  DenseSteer sweep done"
wait $PID_SI 2>/dev/null || true
log "  InFamily sweep done"

# ======== Step 4: Collect and report ========
log "===== Step 4: Collecting results ====="
$PY -u -c "
import json, os
from pathlib import Path

sweep = Path('$SWEEP_DIR')
rows = []
for mode_dir in sorted(sweep.iterdir()):
    if not mode_dir.is_dir():
        continue
    for layer_dir in sorted(mode_dir.iterdir()):
        for lam_dir in sorted(layer_dir.iterdir()):
            for rf in lam_dir.rglob('results_*.json'):
                data = json.load(open(rf))
                r = data.get('results', {})
                if 'logiqa' in r:
                    for mk, mv in r['logiqa'].items():
                        if 'stderr' not in mk and mk != 'alias':
                            rows.append({
                                'mode': mode_dir.name,
                                'layer': layer_dir.name,
                                'lam': lam_dir.name,
                                'metric': mk,
                                'score': mv,
                            })
                            break

baseline = [r for r in rows if r['mode'] == 'BASELINE']
b_score = baseline[0]['score'] if baseline else None
print(f'\nBaseline: {b_score}')

for mode in ['DenseSteer', 'InFamily']:
    mode_rows = sorted([r for r in rows if r['mode'] == mode], key=lambda x: -x['score'])
    if not mode_rows:
        continue
    best = mode_rows[0]
    print(f'\n{mode} best: {best[\"layer\"]} {best[\"lam\"]} score={best[\"score\"]:.4f}')
    if b_score:
        d = (best['score'] - b_score) * 100
        print(f'  delta: {d:+.2f}pp')
    print(f'\n  Top 10:')
    for r in mode_rows[:10]:
        d = (r['score'] - (b_score or 0)) * 100
        print(f'    {r[\"layer\"]:>4} {r[\"lam\"]:>8}  {r[\"score\"]:.4f}  ({d:+.2f}pp)')

with open('$EXPS/sweep_summary.json', 'w') as f:
    json.dump({'baseline': b_score, 'results': rows}, f, indent=2)
print(f'\nSaved sweep_summary.json')
" 2>&1 | tee -a "$LOG"

log "===== PIPELINE COMPLETE ====="
