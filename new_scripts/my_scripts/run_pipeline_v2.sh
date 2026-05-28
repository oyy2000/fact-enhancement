#!/bin/bash
set -euo pipefail

# ============================================================
# LogiQA DenseSteer v2 — Full Pipeline
# Uses test split: back 50 correct for vectors, front 400 for eval
# ============================================================

ROOT="/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement"
PY="/common/users/sl2148/anaconda3/envs/fact_yang/bin/python"
cd "$ROOT"

source .env 2>/dev/null || true
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENAI_BASE_URL="${BASE_URL:-}"

HF_TOKEN_PATH="$ROOT/new_exps/.cache/huggingface/token"
if [ -f "$HF_TOKEN_PATH" ]; then
    export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

EXPS="$ROOT/exps/logiqa_densesteer_v2"
SAMPLE_DIR="$EXPS/samples"
PAIR_DIR="$EXPS/paired"
GPT_DIR="$EXPS/gpt_rewrites"
VEC_DIR="$EXPS/vectors"
SWEEP_DIR="$EXPS/sweep"
LOG="$EXPS/pipeline.log"

mkdir -p "$SAMPLE_DIR" "$PAIR_DIR" "$GPT_DIR" "$VEC_DIR" "$SWEEP_DIR"
> "$LOG"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

JSONL_3B="$SAMPLE_DIR/Qwen2.5-3B_logiqa_test.jsonl"
JSONL_7B="$SAMPLE_DIR/Qwen2.5-7B_logiqa_test.jsonl"
SELECTED_3B="$SAMPLE_DIR/selected_50_3B.json"
SELECTED_7B="$SAMPLE_DIR/selected_50_7B.json"
GPT_JSON="$GPT_DIR/rewritten_old.json"
PAIR_JSON="$PAIR_DIR/paired_3B_7B.json"
DENSE_VEC="$VEC_DIR/N50_dense/Qwen_Qwen2.5-3B-Instruct_applied/steering_vector.pt"
INFAM_VEC="$VEC_DIR/N50_infamily/Qwen_Qwen2.5-3B-Instruct_applied/steering_vector.pt"

GPU="${PIPELINE_GPU:-0}"

# ============================================================
# Step 0: Generate CoT on test split with vLLM (3B then 7B)
# ============================================================
gen_cot() {
    local MODEL="$1" OUT="$2" LABEL="$3"
    if [ -f "$OUT" ] && [ "$(wc -l < "$OUT")" -ge 600 ]; then
        log "Step 0 ($LABEL) SKIP: $OUT has $(wc -l < "$OUT") lines"
        return 0
    fi
    log "Step 0 ($LABEL): Generating CoT on test split..."
    CUDA_VISIBLE_DEVICES=$GPU $PY -u new_scripts/my_scripts/logiqa_generate_cot_vllm.py \
        --model "$MODEL" \
        --split test --limit 651 \
        --out_jsonl "$OUT" 2>&1 | tee -a "$LOG"
    log "Step 0 ($LABEL) DONE: $(wc -l < "$OUT") lines"
}

gen_cot "Qwen/Qwen2.5-3B-Instruct" "$JSONL_3B" "3B"
gen_cot "Qwen/Qwen2.5-7B-Instruct" "$JSONL_7B" "7B"

# ============================================================
# Step 0.5: Select 50 correct samples from back for vectors
# Also create GPT-compatible and pair-compatible JSONs
# ============================================================
log "Step 0.5: Selecting 50 correct samples from back..."
$PY -u -c "
import json, sys
from pathlib import Path

def load_jsonl(p):
    rows = []
    with open(p) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

rows_3b = load_jsonl('$JSONL_3B')
rows_7b = load_jsonl('$JSONL_7B')

# Build 7B lookup by doc_id
map_7b = {}
for r in rows_7b:
    did = r['doc_id']
    resp = r['resps'][0][0] if r.get('resps') and r['resps'][0] else ''
    map_7b[did] = {'resp': resp, 'exact_match': r.get('exact_match', 0)}

# Select 50 correct from back (doc_id descending)
selected_ids = []
for r in reversed(rows_3b):
    if r.get('exact_match', 0) >= 1.0:
        did = r['doc_id']
        if did >= 400:  # only from doc_id >= 400 to not overlap with eval front 400
            selected_ids.append(did)
            if len(selected_ids) >= 50:
                break

print(f'Selected {len(selected_ids)} correct samples (doc_ids: {min(selected_ids)}-{max(selected_ids)})')

if len(selected_ids) < 50:
    print(f'WARNING: Only {len(selected_ids)} correct samples with doc_id>=400, trying doc_id>=350...')
    selected_ids = []
    for r in reversed(rows_3b):
        if r.get('exact_match', 0) >= 1.0 and r['doc_id'] >= 350:
            selected_ids.append(r['doc_id'])
            if len(selected_ids) >= 50:
                break
    print(f'Now have {len(selected_ids)} (doc_ids: {min(selected_ids)}-{max(selected_ids)})')

selected_set = set(selected_ids)

# Build GPT-compatible list (for 00_gpt_modification.py)
gpt_input = []
for r in rows_3b:
    if r['doc_id'] in selected_set:
        resp = r['resps'][0][0] if r.get('resps') and r['resps'][0] else ''
        obj = dict(r)
        obj['resp_before'] = resp
        # doc.question is the formatted problem (for GPT prompt)
        gpt_input.append(obj)

# Save as JSONL for GPT rewrite
gpt_jsonl = '$SAMPLE_DIR/selected_50_3B.jsonl'
with open(gpt_jsonl, 'w') as f:
    for obj in gpt_input:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')
print(f'Wrote {len(gpt_input)} lines to {gpt_jsonl}')

# Build paired JSON for InFamilySteer
paired = []
for r in rows_3b:
    if r['doc_id'] not in selected_set:
        continue
    did = r['doc_id']
    resp_3b = r['resps'][0][0] if r.get('resps') and r['resps'][0] else ''
    info_7b = map_7b.get(did, {})
    resp_7b = info_7b.get('resp', '')
    obj = dict(r)
    obj['resp_before'] = resp_3b
    obj['resp_after'] = resp_7b
    obj['exact_match'] = r.get('exact_match', 0)
    paired.append(obj)

Path('$PAIR_DIR').mkdir(parents=True, exist_ok=True)
with open('$PAIR_JSON', 'w') as f:
    json.dump(paired, f, indent=2, ensure_ascii=False)
print(f'Wrote {len(paired)} paired samples to $PAIR_JSON')

# Stats
n_correct_3b = sum(1 for r in rows_3b if r.get('exact_match', 0) >= 1.0)
n_correct_7b = sum(1 for r in rows_7b if r.get('exact_match', 0) >= 1.0)
print(f'3B correct: {n_correct_3b}/651, 7B correct: {n_correct_7b}/651')
print(f'Pairs with both resp_before != resp_after: {sum(1 for p in paired if p[\"resp_before\"].strip() != p[\"resp_after\"].strip())}')
" 2>&1 | tee -a "$LOG"

# ============================================================
# Step 1a: GPT-5.1 Dense Rewrite (only 50 samples)
# ============================================================
if [ ! -f "$GPT_JSON" ] || [ "$(stat -c%s "$GPT_JSON" 2>/dev/null || echo 0)" -lt 500 ]; then
    if [ -z "$OPENAI_API_KEY" ]; then
        log "Step 1a SKIP: No OPENAI_API_KEY"
    else
        log "Step 1a: GPT-5.1 dense rewrite of 50 selected samples..."
        $PY -u 00_gpt_modification.py \
            --in_jsonl "$SAMPLE_DIR/selected_50_3B.jsonl" \
            --out_json "$GPT_JSON" \
            --prompt_style old \
            --rewrite_last_n 100000 2>&1 | tee -a "$LOG"
        log "Step 1a DONE"
    fi
else
    log "Step 1a SKIP: GPT rewrite exists"
fi

# ============================================================
# Step 2a: Extract DenseSteer vector
# ============================================================
if [ ! -f "$DENSE_VEC" ] && [ -f "$GPT_JSON" ]; then
    log "Step 2a: Extracting DenseSteer vector..."
    CUDA_VISIBLE_DEVICES=$GPU $PY -u new_scripts/my_scripts/logiqa_extract_steering.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --in_path "$GPT_JSON" \
        --out_dir "$VEC_DIR/N50_dense" \
        --num_examples 50 \
        --layers "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18" \
        --tag dense \
        --domain logiqa 2>&1 | tee -a "$LOG"
    log "Step 2a DONE"
elif [ -f "$DENSE_VEC" ]; then
    log "Step 2a SKIP: DenseSteer vector exists"
else
    log "Step 2a SKIP: GPT rewrite missing"
fi

# ============================================================
# Step 2b: Extract InFamily vector
# ============================================================
if [ ! -f "$INFAM_VEC" ] && [ -f "$PAIR_JSON" ]; then
    log "Step 2b: Extracting InFamily vector..."
    CUDA_VISIBLE_DEVICES=$GPU $PY -u new_scripts/my_scripts/logiqa_extract_steering.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --in_path "$PAIR_JSON" \
        --out_dir "$VEC_DIR/N50_infamily" \
        --num_examples 50 \
        --layers "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18" \
        --tag infamily \
        --domain logiqa 2>&1 | tee -a "$LOG"
    log "Step 2b DONE"
elif [ -f "$INFAM_VEC" ]; then
    log "Step 2b SKIP: InFamily vector exists"
else
    log "Step 2b SKIP: pair JSON missing"
fi

# ============================================================
# Step 3: Lambda × Layer sweep on logiqa --limit 400
# ============================================================
log "Step 3: Lambda × Layer sweep (limit=400)"

run_eval() {
    local VEC="$1" LAYER="$2" LAM="$3" TAG="$4"
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
        --limit 400 \
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
        log "  Baseline..."
        run_eval "$ANY_VEC" 0 0.0 BASELINE || true
    fi
fi

# DenseSteer sweep
if [ -f "$DENSE_VEC" ]; then
    log "  DenseSteer sweep..."
    for LAYER in 2 4 6 8 10 12 14 16 18; do
        for LAM in -2.0 -1.0 -0.5 0.5 1.0 2.0 3.0 4.0; do
            run_eval "$DENSE_VEC" $LAYER $LAM DenseSteer || true
        done
    done
    log "  DenseSteer sweep DONE"
fi

# InFamily sweep
if [ -f "$INFAM_VEC" ]; then
    log "  InFamily sweep..."
    for LAYER in 2 4 6 8 10 12 14 16 18; do
        for LAM in -1.0 -0.5 -0.3 0.3 0.45 0.5 0.7 1.0; do
            run_eval "$INFAM_VEC" $LAYER $LAM InFamily || true
        done
    done
    log "  InFamily sweep DONE"
fi

# ============================================================
# Step 4: Collect results
# ============================================================
log "Step 4: Collecting results"
$PY -u -c "
import json
from pathlib import Path

sweep = Path('$SWEEP_DIR')
rows = []
for mode_dir in sorted(sweep.iterdir()):
    if not mode_dir.is_dir(): continue
    for layer_dir in sorted(mode_dir.iterdir()):
        if not layer_dir.is_dir(): continue
        for lam_dir in sorted(layer_dir.iterdir()):
            if not lam_dir.is_dir(): continue
            for rf in lam_dir.rglob('results_*.json'):
                data = json.load(open(rf))
                r = data.get('results', {})
                if 'logiqa' in r:
                    rows.append({
                        'mode': mode_dir.name,
                        'layer': layer_dir.name,
                        'lam': lam_dir.name,
                        'acc': r['logiqa'].get('acc,none', 0),
                        'acc_norm': r['logiqa'].get('acc_norm,none', 0),
                    })
                    break

baseline = [r for r in rows if r['mode'] == 'BASELINE']
b_norm = baseline[0]['acc_norm'] if baseline else None
b_acc = baseline[0]['acc'] if baseline else None
print(f'Baseline: acc={b_acc:.4f}  acc_norm={b_norm:.4f}  (limit=400)')
print()

for mode in ['DenseSteer', 'InFamily']:
    mode_rows = sorted([r for r in rows if r['mode'] == mode], key=lambda x: -x['acc_norm'])
    if not mode_rows: continue
    best = mode_rows[0]
    d = (best['acc_norm'] - b_norm) * 100 if b_norm else 0
    print(f'{mode} best: {best[\"layer\"]} {best[\"lam\"]}  acc_norm={best[\"acc_norm\"]:.4f}  (Δ={d:+.2f}pp)')
    print(f'  Top 10:')
    for r in mode_rows[:10]:
        d = (r['acc_norm'] - (b_norm or 0)) * 100
        print(f'    {r[\"layer\"]:>4} {r[\"lam\"]:>10}  acc={r[\"acc\"]:.4f}  acc_norm={r[\"acc_norm\"]:.4f}  ({d:+.2f}pp)')
    print()

with open('$EXPS/sweep_summary.json', 'w') as f:
    json.dump({'baseline_acc': b_acc, 'baseline_acc_norm': b_norm, 'results': rows}, f, indent=2)
print(f'Total runs: {len(rows)}')
print(f'Saved $EXPS/sweep_summary.json')
" 2>&1 | tee -a "$LOG"

log "===== PIPELINE v2 COMPLETE ====="
