#!/usr/bin/env bash
set -euo pipefail

# Parallel DAS scoring for Figure 1 GSM8K sampling data.
#
# Default behavior:
#   - uses 7 GPUs: 0,1,2,3,4,5,6
#   - launches one process per Qwen model folder
#   - each process loads the fixed 3B DAS scorer once on its assigned GPU
#   - writes gsm8k_samples_with_das.jsonl inside each model folder
#
# Examples:
#   bash new_scripts/run_add_das_parallel_7gpu.sh
#   GPU_LIST=1,2,3,4,5,6,7 BATCH_SIZE=16 bash new_scripts/run_add_das_parallel_7gpu.sh
#   FORCE=1 bash new_scripts/run_add_das_parallel_7gpu.sh

ROOT="/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement"
PYTHON="${PYTHON:-/common/users/sl2148/anaconda3/envs/fact_yang/bin/python3}"
SCRIPT="$ROOT/new_scripts/add_das_to_figure1_sampling.py"
DATA_DIR="${DATA_DIR:-$ROOT/new_exps/figure1_sampling_data}"
LOG_DIR="${LOG_DIR:-$DATA_DIR/das_logs}"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
DTYPE="${DTYPE:-float16}"
SCORER_MODEL="${SCORER_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
INPUT_NAME="${INPUT_NAME:-gsm8k_samples.jsonl}"
OUTPUT_NAME="${OUTPUT_NAME:-gsm8k_samples_with_das.jsonl}"
FORCE="${FORCE:-0}"

MODELS=(
  "Qwen_Qwen2.5-0.5B-Instruct"
  "Qwen_Qwen2.5-1.5B-Instruct"
  "Qwen_Qwen2.5-3B-Instruct"
  "Qwen_Qwen2.5-7B-Instruct"
  "Qwen_Qwen2.5-14B-Instruct"
  "Qwen_Qwen2.5-32B-Instruct"
  "Qwen_Qwen2.5-72B-Instruct"
)

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if (( ${#GPUS[@]} < ${#MODELS[@]} )); then
  echo "Need at least ${#MODELS[@]} GPUs for this launcher, got ${#GPUS[@]} from GPU_LIST=$GPU_LIST" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
cd "$ROOT"

echo "DAS parallel scoring"
echo "  scorer:      $SCORER_MODEL"
echo "  gpu list:    $GPU_LIST"
echo "  batch size:  $BATCH_SIZE"
echo "  max length:  $MAX_LENGTH"
echo "  dtype:       $DTYPE"
echo "  output name: $OUTPUT_NAME"
echo "  logs:        $LOG_DIR"
echo

pids=()
launched=0

for i in "${!MODELS[@]}"; do
  model="${MODELS[$i]}"
  gpu="${GPUS[$i]}"
  out_path="$DATA_DIR/$model/$OUTPUT_NAME"
  log_path="$LOG_DIR/${model}.das.log"

  if [[ "$FORCE" != "1" && -s "$out_path" ]]; then
    echo "[skip] $model already has $out_path (set FORCE=1 to overwrite)"
    continue
  fi

  echo "[launch] GPU $gpu -> $model"
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON" "$SCRIPT" \
      --data-dir "$DATA_DIR" \
      --input-name "$INPUT_NAME" \
      --output-name "$OUTPUT_NAME" \
      --scorer-model "$SCORER_MODEL" \
      --models "$model" \
      --device-map single \
      --batch-size "$BATCH_SIZE" \
      --max-length "$MAX_LENGTH" \
      --dtype "$DTYPE"
  ) > "$log_path" 2>&1 &

  pids+=("$!")
  launched=$((launched + 1))
done

if (( launched == 0 )); then
  echo "Nothing to do."
  exit 0
fi

echo
echo "Launched $launched jobs. Waiting..."

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

echo
if (( failed )); then
  echo "One or more DAS scoring jobs failed. Check logs in $LOG_DIR" >&2
  exit 1
fi

echo "All DAS scoring jobs finished."
for model in "${MODELS[@]}"; do
  out_path="$DATA_DIR/$model/$OUTPUT_NAME"
  if [[ -s "$out_path" ]]; then
    echo "  ok: $out_path"
  else
    echo "  missing: $out_path"
  fi
done
