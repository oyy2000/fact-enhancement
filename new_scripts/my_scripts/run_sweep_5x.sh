#!/bin/bash
set -uo pipefail

ROOT="/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement"
PY="/common/users/sl2148/anaconda3/envs/fact_yang/bin/python"
cd "$ROOT"

HF_TOKEN_PATH="$ROOT/new_exps/.cache/huggingface/token"
if [ -f "$HF_TOKEN_PATH" ]; then
    export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

SWEEP_DIR="$ROOT/exps/logiqa_densesteer_v2/sweep_5x"
DENSE_VEC="$ROOT/exps/logiqa_densesteer_v2/vectors/N50_dense/Qwen_Qwen2.5-3B-Instruct_applied/steering_vector.pt"
INFAM_VEC="$ROOT/exps/logiqa_densesteer_v2/vectors/N50_infamily/Qwen_Qwen2.5-3B-Instruct_applied/steering_vector.pt"

mkdir -p "$SWEEP_DIR"

GPU_D="${1:-0}"
GPU_I="${2:-6}"

run_eval() {
    local VEC="$1" LAYER="$2" LAM="$3" TAG="$4" GPU="$5"
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

echo "[$(date '+%H:%M:%S')] Starting 5x sweep"

# DenseSteer on GPU_D
sweep_dense() {
    for LAYER in 2 4 6 8 10 12 14 16 18; do
        for LAM in -10.0 -5.0 -2.5 2.5 5.0 10.0 15.0 20.0; do
            run_eval "$DENSE_VEC" $LAYER $LAM DenseSteer $GPU_D || true
        done
    done
    echo "[$(date '+%H:%M:%S')] DenseSteer 5x sweep DONE"
}

# InFamily on GPU_I
sweep_infamily() {
    for LAYER in 2 4 6 8 10 12 14 16 18; do
        for LAM in -5.0 -2.5 -1.5 1.5 2.25 2.5 3.5 5.0; do
            run_eval "$INFAM_VEC" $LAYER $LAM InFamily $GPU_I || true
        done
    done
    echo "[$(date '+%H:%M:%S')] InFamily 5x sweep DONE"
}

# Run both in parallel on different GPUs
sweep_dense &
PID_D=$!
sweep_infamily &
PID_I=$!

echo "DenseSteer PID=$PID_D (GPU $GPU_D), InFamily PID=$PID_I (GPU $GPU_I)"
wait $PID_D 2>/dev/null || true
wait $PID_I 2>/dev/null || true

echo "[$(date '+%H:%M:%S')] All 5x sweeps done"

# Collect results (merge with original sweep)
$PY -u -c "
import json
from pathlib import Path

rows = []
for sweep_dir in ['exps/logiqa_densesteer_v2/sweep', 'exps/logiqa_densesteer_v2/sweep_5x']:
    sweep = Path(sweep_dir)
    if not sweep.exists(): continue
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
                        layer = int(layer_dir.name[1:])
                        lam = float(lam_dir.name.split('_')[1])
                        rows.append({
                            'mode': mode_dir.name,
                            'layer': layer,
                            'lam': lam,
                            'acc': r['logiqa'].get('acc,none', 0),
                            'acc_norm': r['logiqa'].get('acc_norm,none', 0),
                            'sweep': sweep_dir.split('/')[-1],
                        })
                        break

baseline = [r for r in rows if r['mode'] == 'BASELINE']
b_norm = baseline[0]['acc_norm'] if baseline else None
b_acc = baseline[0]['acc'] if baseline else None
print(f'Baseline: acc={b_acc:.4f}  acc_norm={b_norm:.4f}')
print()

for mode in ['DenseSteer', 'InFamily']:
    mode_rows = sorted([r for r in rows if r['mode'] == mode], key=lambda x: -x['acc_norm'])
    if not mode_rows: continue
    best = mode_rows[0]
    d = (best['acc_norm'] - b_norm) * 100 if b_norm else 0
    print(f'{mode} best: L{best[\"layer\"]} λ={best[\"lam\"]}  acc_norm={best[\"acc_norm\"]:.4f}  (Δ={d:+.2f}pp)  [{best[\"sweep\"]}]')
    print(f'  Top 15:')
    for r in mode_rows[:15]:
        d = (r['acc_norm'] - (b_norm or 0)) * 100
        print(f'    L{r[\"layer\"]:>2} λ={r[\"lam\"]:>6}  acc={r[\"acc\"]:.4f}  acc_norm={r[\"acc_norm\"]:.4f}  ({d:+.2f}pp)  [{r[\"sweep\"]}]')
    print()

with open('exps/logiqa_densesteer_v2/sweep_5x_summary.json', 'w') as f:
    json.dump({'baseline_acc': b_acc, 'baseline_acc_norm': b_norm, 'results': rows}, f, indent=2)
print(f'Total runs: {len(rows)}')
"
