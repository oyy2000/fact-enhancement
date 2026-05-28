#!/usr/bin/env bash
# Non-math reasoning steering: LogiQA (default) or HotpotQA fallback (CALIB_DATASET=hotpotqa).
set -euo pipefail

ROOT="/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement"
PY="${PY:-/common/users/sl2148/anaconda3/envs/fact_yang/bin/python}"
cd "$ROOT"

LIMIT="${LIMIT:-800}"
GPU3B="${GPU3B:-0}"
GPU7B="${GPU7B:-1}"
CALIB_DATASET="${CALIB_DATASET:-logiqa}"

if [[ "$CALIB_DATASET" == "hotpotqa" ]]; then
  SAMPLE_DIR="$ROOT/exps/hotpotqa_densesteer/samples"
  PAIR_DIR="$ROOT/exps/hotpotqa_densesteer/paired"
  GPT_DIR="$ROOT/exps/hotpotqa_densesteer/gpt_rewrites/Qwen_Qwen2.5-3B-Instruct"
  PAIR_JSON="$PAIR_DIR/Qwen3B_paired_Qwen7B_hotpotqa.json"
  JSONL_3B="$SAMPLE_DIR/Qwen_Qwen2.5-3B-Instruct_hotpotqa_cot.jsonl"
  JSONL_7B="$SAMPLE_DIR/Qwen_Qwen2.5-7B-Instruct_hotpotqa_cot.jsonl"
  GEN_LIMIT="${HOTPOT_LIMIT:-200}"
else
  SAMPLE_DIR="$ROOT/exps/logiqa_densesteer/samples"
  PAIR_DIR="$ROOT/exps/logiqa_densesteer/paired"
  GPT_DIR="$ROOT/exps/logiqa_densesteer/gpt_rewrites/Qwen_Qwen2.5-3B-Instruct"
  PAIR_JSON="$PAIR_DIR/Qwen3B_paired_Qwen7B_logiqa.json"
  JSONL_3B="$SAMPLE_DIR/Qwen_Qwen2.5-3B-Instruct_logiqa_cot_train.jsonl"
  JSONL_7B="$SAMPLE_DIR/Qwen_Qwen2.5-7B-Instruct_logiqa_cot_train.jsonl"
  GEN_LIMIT="$LIMIT"
fi

GPT_JSON="$GPT_DIR/rewritten_old.json"
mkdir -p "$SAMPLE_DIR" "$PAIR_DIR" "$GPT_DIR"

echo "CALIB_DATASET=$CALIB_DATASET (set CALIB_DATASET=hotpotqa if LogiQA unavailable)"

if [[ "$CALIB_DATASET" == "hotpotqa" ]]; then
  echo "== [0a] HotpotQA CoT 3B (GPU $GPU3B) =="
  CUDA_VISIBLE_DEVICES="$GPU3B" "$PY" new_scripts/my_scripts/hotpotqa_generate_cot.py \
    --model "Qwen/Qwen2.5-3B-Instruct" --split train --limit "$GEN_LIMIT" \
    --out_jsonl "$JSONL_3B"

  echo "== [0b] HotpotQA CoT 7B (GPU $GPU7B) =="
  CUDA_VISIBLE_DEVICES="$GPU7B" "$PY" new_scripts/my_scripts/hotpotqa_generate_cot.py \
    --model "Qwen/Qwen2.5-7B-Instruct" --split train --limit "$GEN_LIMIT" \
    --out_jsonl "$JSONL_7B"
else
  echo "== [0a] LogiQA CoT 3B (GPU $GPU3B) =="
  CUDA_VISIBLE_DEVICES="$GPU3B" "$PY" new_scripts/my_scripts/logiqa_generate_cot.py \
    --model "Qwen/Qwen2.5-3B-Instruct" --split train --limit "$GEN_LIMIT" \
    --out_jsonl "$JSONL_3B"

  echo "== [0b] LogiQA CoT 7B (GPU $GPU7B) =="
  CUDA_VISIBLE_DEVICES="$GPU7B" "$PY" new_scripts/my_scripts/logiqa_generate_cot.py \
    --model "Qwen/Qwen2.5-7B-Instruct" --split train --limit "$GEN_LIMIT" \
    --out_jsonl "$JSONL_7B"
fi

echo "== [1a] Pair 3B/7B for InFamilySteer =="
"$PY" new_scripts/my_scripts/logiqa_pair_infamily.py \
  --small_jsonl "$JSONL_3B" --large_jsonl "$JSONL_7B" \
  --out_json "$PAIR_JSON" \
  --small_model "Qwen/Qwen2.5-3B-Instruct" \
  --large_model "Qwen/Qwen2.5-7B-Instruct"

echo "== [1b] GPT dense rewrite (needs OPENAI_API_KEY) =="
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "SKIP GPT rewrite: set OPENAI_API_KEY then run:"
  echo "  $PY 00_gpt_modification.py --in_jsonl \"$JSONL_3B\" --out_json \"$GPT_JSON\" --prompt_style old --rewrite_last_n 100000"
else
  "$PY" 00_gpt_modification.py \
    --in_jsonl "$JSONL_3B" \
    --out_json "$GPT_JSON" \
    --prompt_style old \
    --rewrite_last_n 100000

  echo "== [2a] Extract DenseSteer vector =="
  "$PY" new_scripts/my_scripts/logiqa_extract_steering.py \
    --model "Qwen/Qwen2.5-3B-Instruct" \
    --in_path "$GPT_JSON" \
    --num_examples 50 \
    --layers 6 \
    --tag dense_gpt_old \
    --domain "${CALIB_DATASET}"
fi

echo "== [2b] Extract InFamilySteer vector =="
"$PY" new_scripts/my_scripts/logiqa_extract_steering.py \
  --model "Qwen/Qwen2.5-3B-Instruct" \
  --in_path "$PAIR_JSON" \
  --num_examples 50 \
  --layers 6 \
  --tag infamily_7b \
  --domain "${CALIB_DATASET}"

echo "Done. Calibrate then E11/E12:"
if [[ "$CALIB_DATASET" == "hotpotqa" ]]; then
  echo "  $PY new_scripts/my_scripts/e11_e12_logiqa_steer_exp.py --steering_suite hotpotqa --experiments CALIB_HOTPOTQA --batch_size 1 --gpus 0"
  echo "  $PY new_scripts/my_scripts/e11_e12_logiqa_steer_exp.py --steering_suite hotpotqa --experiments E11 E12 --batch_size 16 --gpus 0 1 2 3 4 5 6 7"
else
  echo "  $PY new_scripts/my_scripts/e11_e12_logiqa_steer_exp.py --steering_suite logiqa --experiments CALIB_LOGIQA --gpus 0"
  echo "  $PY new_scripts/my_scripts/e11_e12_logiqa_steer_exp.py --steering_suite logiqa --experiments E11 E12 --gpus 0 1 2 3 4 5 6 7"
fi
