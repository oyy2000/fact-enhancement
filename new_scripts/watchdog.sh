#!/bin/bash
# Watchdog: waits, then loops submit+monitor until all non-skip tasks are done.
# Usage: nohup bash new_scripts/watchdog.sh > new_exps/watchdog.log 2>&1 &

set -uo pipefail

PY=/common/home/sl2148/anaconda3/envs/fact_yang/bin/python
BASE=/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement
cd "$BASE"

DELAY_SEC=${1:-7200}  # default 2 hours
MAX_ROUNDS=20
ROUND_WAIT=300        # 5 min between rounds

echo "==========================================="
echo " Watchdog started at $(date)"
echo " Delay: ${DELAY_SEC}s before first run"
echo " Max rounds: $MAX_ROUNDS"
echo "==========================================="

if [ "$DELAY_SEC" -gt 0 ]; then
    echo "Sleeping ${DELAY_SEC}s (until $(date -d "+${DELAY_SEC} seconds" '+%H:%M:%S %Y-%m-%d'))..."
    sleep "$DELAY_SEC"
    echo "Woke up at $(date)"
fi

for round in $(seq 1 $MAX_ROUNDS); do
    echo ""
    echo "========== Round $round / $MAX_ROUNDS  $(date) =========="

    # Kill any stale GPU processes from our scripts
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read pid; do
        cmdline=$(ps -o args= -p "$pid" 2>/dev/null || echo "")
        if echo "$cmdline" | grep -q "figure1_"; then
            echo "  Killing stale process $pid"
            kill "$pid" 2>/dev/null
        fi
    done
    sleep 5

    # Check status
    STATUS=$($PY -u new_scripts/run_pending.py --status 2>&1)
    echo "$STATUS"

    # Extract counts
    DONE=$(echo "$STATUS" | grep "Summary:" | grep -oP '\d+ done' | grep -oP '\d+')
    PENDING=$(echo "$STATUS" | grep "Summary:" | grep -oP '\d+ pending' | grep -oP '\d+')
    FAILED=$(echo "$STATUS" | grep "Summary:" | grep -oP '\d+ failed' | grep -oP '\d+')
    TOTAL=$((DONE + PENDING + FAILED))

    echo "  Parsed: done=$DONE pending=$PENDING failed=$FAILED"

    if [ "${PENDING:-0}" -eq 0 ] && [ "${FAILED:-0}" -eq 0 ]; then
        echo ""
        echo "All non-skip tasks complete!"
        break
    fi

    # Find free GPUs (>38GB free)
    FREE_GPUS=$($PY -c "
import subprocess
out = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'], text=True)
gpus = []
for line in out.strip().split('\n'):
    idx, mem = line.split(',')
    if int(mem.strip()) >= 38000:
        gpus.append(idx.strip())
print(','.join(gpus) if gpus else '')
" 2>/dev/null)

    if [ -z "$FREE_GPUS" ]; then
        echo "  No free GPUs (>=38GB). Waiting ${ROUND_WAIT}s..."
        sleep "$ROUND_WAIT"
        continue
    fi

    echo "  Free GPUs: $FREE_GPUS"
    echo "  Submitting with --retry-failed..."

    # Run scheduler in foreground (blocks until all scheduled tasks finish)
    $PY -u new_scripts/run_pending.py --retry-failed --gpus "$FREE_GPUS" 2>&1

    echo "  Round $round scheduler finished at $(date)"
    sleep 30  # brief cooldown
done

echo ""
echo "==========================================="
echo " Watchdog final status at $(date)"
echo "==========================================="
$PY -u new_scripts/run_pending.py --status 2>&1

# Auto-fill Excel with any new results
echo ""
echo "Updating Excel with completed results..."
$PY -u << 'PYEOF'
import json, os, numpy as np, openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

BASE = '/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement'
XLSX = os.path.join(BASE, 'documents/rebuttal_experiment_data.xlsx')
EXPECTED = {"gsm8k": 1319, "math500": 500, "aime": 30, "amc": 40, "olympiad": 675}

def get_path(model, dataset):
    ms = model.replace("/", "_")
    if dataset == "gsm8k":
        return os.path.join(BASE, "new_exps/figure1_sampling_data", ms, "gsm8k_samples.jsonl")
    return os.path.join(BASE, "new_exps/figure1_multi_dataset", dataset, ms, "samples.jsonl")

def extract_stats(model, dataset):
    p = get_path(model, dataset)
    if not os.path.isfile(p):
        return None
    lines = open(p).readlines()
    if len(lines) < EXPECTED[dataset]:
        return None
    all_correct, all_steps, all_density, all_tokens = [], [], [], []
    correct_steps, incorrect_steps, correct_density, incorrect_density = [], [], [], []
    n_correct, n_incorrect = 0, 0
    for line in lines:
        doc = json.loads(line)
        for s in doc["samples"]:
            c = s["correct"]
            st = s["n_steps"]
            rho = s.get("density_rho", s.get("avg_tokens_per_step", 0))
            tok = s["total_tokens"]
            all_correct.append(int(c))
            all_steps.append(st)
            all_density.append(rho)
            all_tokens.append(tok)
            if c:
                n_correct += 1; correct_steps.append(st); correct_density.append(rho)
            else:
                n_incorrect += 1; incorrect_steps.append(st); incorrect_density.append(rho)
    acc_std = np.std([np.mean([s["correct"] for s in json.loads(l)["samples"]]) for l in lines]) * 100
    return {
        "n_questions": len(lines), "n_samples": len(all_correct),
        "accuracy": round(np.mean(all_correct)*100, 2), "acc_std": round(acc_std, 2),
        "avg_steps": round(np.mean(all_steps), 2), "std_steps": round(np.std(all_steps), 2),
        "avg_density": round(np.mean(all_density), 2), "std_density": round(np.std(all_density), 2),
        "avg_tokens": round(np.mean(all_tokens), 2),
        "n_correct": n_correct, "n_incorrect": n_incorrect,
        "correct_avg_steps": round(np.mean(correct_steps), 2) if correct_steps else 0,
        "incorrect_avg_steps": round(np.mean(incorrect_steps), 2) if incorrect_steps else 0,
        "correct_avg_density": round(np.mean(correct_density), 2) if correct_density else 0,
        "incorrect_avg_density": round(np.mean(incorrect_density), 2) if incorrect_density else 0,
    }

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct","Qwen/Qwen2.5-1.5B-Instruct","Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct","Qwen/Qwen2.5-14B-Instruct","Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct","meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct","meta-llama/Llama-3.1-8B-Instruct",
]
DATASETS = ["gsm8k","math500","aime","amc","olympiad"]
DS_MAP = {"GSM8K":"gsm8k","MATH-500":"math500","AIME":"aime","AMC":"amc","Olympiad":"olympiad"}

results = {}
for model in MODELS:
    short = model.split("/")[-1].replace("-Instruct","")
    for ds in DATASETS:
        stats = extract_stats(model, ds)
        if stats:
            results[f"{short}|{ds}"] = stats

wb = openpyxl.load_workbook(XLSX)
green = PatternFill(start_color="D5F5D5", end_color="D5F5D5", fill_type="solid")
border = Border(left=Side(style='thin'),right=Side(style='thin'),top=Side(style='thin'),bottom=Side(style='thin'))
center = Alignment(horizontal='center', wrap_text=True)

def norm(raw):
    return str(raw).strip().replace("-Instruct","")

updated = 0
ws = wb['E4 Fig1 Scaling']
for row in range(2, ws.max_row+1):
    mc, dc = ws.cell(row=row,column=1).value, ws.cell(row=row,column=3).value
    if not mc or not dc: continue
    key = f"{norm(mc)}|{DS_MAP.get(dc, dc.lower())}"
    if key not in results: continue
    if ws.cell(row=row,column=6).value not in (None, "TODO"): continue
    r = results[key]
    for col, val in [(4,r["n_questions"]),(5,r["n_samples"]),(6,r["accuracy"]),(7,r["acc_std"]),(8,r["avg_steps"]),(9,r["std_steps"]),(10,r["avg_density"]),(11,r["std_density"]),(12,r["avg_tokens"]),(13,"DONE")]:
        c = ws.cell(row=row,column=col,value=val); c.fill=green; c.font=Font(); c.border=border; c.alignment=center
    updated += 1

ws56 = wb['E5-E6 Density Analysis']
for row in range(2, ws56.max_row+1):
    mc, dc = ws56.cell(row=row,column=1).value, ws56.cell(row=row,column=2).value
    if not mc or not dc: continue
    key = f"{norm(mc)}|{DS_MAP.get(dc, dc.lower())}"
    if key not in results: continue
    if ws56.cell(row=row,column=3).value not in (None, "TODO"): continue
    r = results[key]
    ds = round(r["correct_avg_steps"]-r["incorrect_avg_steps"],2)
    dd = round(r["correct_avg_density"]-r["incorrect_avg_density"],2)
    for col, val in [(3,r["n_correct"]),(4,r["n_incorrect"]),(5,r["correct_avg_steps"]),(6,r["incorrect_avg_steps"]),(7,ds),(8,r["correct_avg_density"]),(9,r["incorrect_avg_density"]),(10,dd),(11,r["avg_density"]),(12,r["accuracy"])]:
        c = ws56.cell(row=row,column=col,value=val); c.fill=green; c.font=Font(); c.border=border; c.alignment=center
    updated += 1

ws_sm = wb['Status Matrix']
col_ds = {2:"gsm8k",3:"math500",4:"aime",5:"amc",6:"olympiad"}
for row in range(2, ws_sm.max_row+1):
    mc = ws_sm.cell(row=row,column=1).value
    if not mc or "Done" in str(mc) or "Family" in str(mc): continue
    done_ct = 0
    for col, dsk in col_ds.items():
        if f"{str(mc).strip()}|{dsk}" in results:
            ws_sm.cell(row=row,column=col,value="DONE").fill = green
            done_ct += 1
        elif ws_sm.cell(row=row,column=col).value == "DONE":
            done_ct += 1
    ws_sm.cell(row=row,column=7,value=f"{done_ct}/5")

wb.save(XLSX)
print(f"Excel updated: {updated} cells filled. Saved to {XLSX}")
PYEOF

echo ""
echo "==========================================="
echo " Watchdog finished at $(date)"
echo "==========================================="
