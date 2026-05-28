#!/usr/bin/bash
# Queue NLL-over-generation for E0–E3.2 (one GPU per experiment, up to 8 GPUs).
# Uses best GSM8K sweep sample jsonl per control (same criterion as fill_e0_e3_control_tokens_rho.py).
#
# Usage:
#   bash new_scripts/submit_control_generation_nll.sh
#   bash new_scripts/submit_control_generation_nll.sh --only-correct   # GSM8K exact_match==1 only
#   bash new_scripts/submit_control_generation_nll.sh --dry-run
#   GPU_LIST=0,1,2,3,4 bash new_scripts/submit_control_generation_nll.sh --only-correct
#
set -euo pipefail
ROOT="/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement"
cd "$ROOT"

PY="${PY:-/common/home/sl2148/anaconda3/envs/fact_yang/bin/python}"
SCRIPT="$ROOT/new_scripts/my_scripts/control_generation_nll.py"
LOGDIR="$ROOT/control_experiments/Qwen_Qwen2.5-3B-Instruct/nll_generation_logs"
mkdir -p "$LOGDIR"

EXPS=(e0 e1 e2 e3 e3.2)
if [[ -n "${GPU_LIST:-}" ]]; then
  IFS=',' read -r -a GPUS <<< "$GPU_LIST"
else
  GPUS=(0 1 2 3 4 5 6 7)
fi

DRY=0
EXTRA=()
for a in "$@"; do
  if [[ "$a" == "--dry-run" ]]; then DRY=1
  else EXTRA+=("$a"); fi
done

n_exp=${#EXPS[@]}
n_gpu=${#GPUS[@]}
if (( n_gpu < n_exp )); then
  echo "Need at least $n_exp GPUs in GPU_LIST (got $n_gpu). Set GPU_LIST=0,1,2,3,4" >&2
  exit 1
fi

echo "Python: $PY"
echo "Log dir: $LOGDIR"
echo "Experiments: ${EXPS[*]}"
echo "GPUs: ${GPUS[*]}"

for i in "${!EXPS[@]}"; do
  exp="${EXPS[$i]}"
  gpu="${GPUS[$i]}"
  ts="$(date +%Y%m%d_%H%M%S)"
  log="$LOGDIR/${exp}_gpu${gpu}_${ts}.log"
  if [[ "$DRY" == 1 ]]; then
    echo "[dry-run] CUDA_VISIBLE_DEVICES=$gpu $PY -u $SCRIPT --experiment $exp ${EXTRA[*]} > $log"
  else
    echo "[submit] $exp on GPU $gpu -> $log"
    CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" -u "$SCRIPT" --experiment "$exp" "${EXTRA[@]}" >"$log" 2>&1 &
    echo $! >"$LOGDIR/${exp}_gpu${gpu}_${ts}.pid"
  fi
done

if [[ "$DRY" == 0 ]]; then
  echo "All jobs started in background. tail -f $LOGDIR/*.log"
fi
