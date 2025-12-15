import json, torch
import numpy as np
import torch.nn.functional as F
from scipy.stats import pearsonr
from transformers import AutoTokenizer, AutoModel
import os
import matplotlib.pyplot as plt
PRM_MODEL="Qwen/Qwen2.5-Math-PRM-7B"
from datetime import datetime


tokenizer=AutoTokenizer.from_pretrained(PRM_MODEL,trust_remote_code=True)
model=AutoModel.from_pretrained(
    PRM_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
).eval()

STEP_TOKEN = tokenizer.encode("<extra_0>")[0]

def avg_step_length(steps):
    return np.mean([len(s.split()) for s in steps]) if len(steps)>0 else 0


# ---------------- Step Scoring ----------------
def prm_step_scores(logits, input_ids):
    probs = F.softmax(logits, dim=-1)
    idx   = (input_ids==STEP_TOKEN).nonzero(as_tuple=True)[1]
    return [ probs[0, idx[i]:idx[i+1], 1].mean().item()
             for i in range(len(idx)-1) ]


def eval_cot_prm(system, query, steps):
    text = "<extra_0>".join(steps) + "<extra_0>"

    msgs = [
        {"role":"system", "content":system},
        {"role":"user",   "content":query},
        {"role":"assistant", "content":text}
    ]

    conv = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    ids  = tokenizer.encode(conv, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)

    return prm_step_scores(out[0], ids)

def analyze_step_errors(scores, thr=0.5):
    """
    Returns:
        earliest_err  : index of first low-confidence step  (1-based)
        prefix_len    : how many steps are consecutively correct from the start
    """
    n = len(scores)

    # ---------- earliest error step (first score<thr) ----------
    err_idx = next((i+1 for i,s in enumerate(scores) if s < thr), None)

    # ---------- prefix correctness ----------
    prefix_len = 0
    for s in scores:
        if s > thr: prefix_len += 1
        else: break

    return err_idx, prefix_len
    

def run_dataset(jsonl, thr=0.5, label="exact_match"):
    data=[json.loads(l) for l in open(jsonl)]

    F_full=[];F_no_last=[];F_hard=[];F_hard_no_last=[];
    EARLY=[]; PREFIX=[]; AVG_LEN=[]; STEPS=[];
    Y=[]

    for i,d in enumerate(data):
        if d.get("filter") == "strict-match": continue

        cot = d["resps"][0][0].strip()
        steps = [s.strip() for s in cot.split("\n") if s.strip()] \
                if "\n" in cot else \
                [s.strip() for s in cot.split(".") if s.strip()]

        if len(steps)==0: continue  # skip broken samples
        scores = eval_cot_prm("", d["arguments"]["gen_args_0"]["arg_0"], steps)

        Fi_full = sum(scores)/len(scores)
        Fi_no_last = sum(scores[:-1])/len(scores[:-1]) if len(scores)>1 else Fi_full
        Fi_hard = sum(s>thr for s in scores)/len(scores)
        Fi_hard_no_last = sum(s>thr for s in scores[:-1])/(len(scores)-1) if len(scores)>1 else Fi_hard

        earliest_err, prefix_len = analyze_step_errors(scores, thr)
        avg_len = avg_step_length(steps)   # ⭐ new

        yi = int(d.get(label,0))

        print(f"[{i}] steps={len(scores)} | avg_step_len={avg_len:.1f} | scores={scores} | len(steps)={len(steps)}" )
        print(f"  Full={Fi_full:.3f} NoLast={Fi_no_last:.3f} | Hard={Fi_hard:.3f} HardNL={Fi_hard_no_last:.3f}")
        print(f"  🔸EarliestError={earliest_err} 🔸PrefixOK={prefix_len} 🔸AvgLen={avg_len} | Y={yi}\n")

        F_full.append(Fi_full);F_no_last.append(Fi_no_last)
        F_hard.append(Fi_hard);F_hard_no_last.append(Fi_hard_no_last)
        EARLY.append(earliest_err);PREFIX.append(prefix_len);AVG_LEN.append(avg_len)
        STEPS.append(len(steps))
        Y.append(yi)

    # safe corr
    def c(a): return pearsonr(a,Y)[0] if len(a)>1 else 0

    print("\n━━━━ Aggregate ━━━━")
    print(f"Step length mean = {np.mean(AVG_LEN):.2f}")
    print("━━━━━━━━━━━━━━━━━━\n")

    return F_full, F_no_last, F_hard, F_hard_no_last, EARLY, PREFIX, AVG_LEN, STEPS, Y


import glob, json, os, numpy as np
# ===================== CONFIG =====================
BASE_DIR = "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid/gsm8k_cot_zeroshot"
lam_values = ["lam-0p5","lam-1p0","lam-1p5","lam-2p0", "BASELINE","lam0p5","lam1p0","lam1p5","lam2p0"]   # 横轴维度

# λ label 显示用
lam_plot_labels = {
    "lam-2p0":"-2.0",
    "lam-1p5":"-1.5",
    "lam-1p0":"-1.0",
    "lam-0p5":"-0.5",
    "BASELINE":"0.0",
    "lam0p5":"0.5",
    "lam1p0":"1.0",
    "lam1p5":"1.5",
    "lam2p0":"2.0"
}

L_values   = ["L24"]
SAVE_PATH = f"prm_results_live_dump_{datetime.now().strftime('%Y%m%d')}.json"   # 你可以换路径
# ==================================================
model_names = {"Llama-3.1-8B-Instruct": "meta-llama__Llama-3.1-8B-Instruct"}
        #    "Qwen2.5-7B-Instruct": "Qwen__Qwen2.5-7B-Instruct"}
results = {}

for model_name in model_names:
    results.setdefault(model_name, {})

    for L in L_values:
        results[model_name].setdefault(L, {})

        for lam in lam_values:
            folder = f"{BASE_DIR}/{model_name}_{L}_{lam}/{model_names[model_name]}/"
            pattern = os.path.join(folder, "samples_gsm8k_cot_zeroshot_*.jsonl")
            print(folder)
            files = sorted(glob.glob(pattern))  # <-- Only matching correct jsonl names
            
            if len(files)==0:
                print(f"⚠ No jsonl found for {L}-{lam}")
                continue
            
            # ⭐ Pick newest file
            jsonl = files[-1]
            print(f"🚀 Running {L}-{lam}  |  using latest JSONL:\n → {jsonl}\n")

            F_full, F_no_last, F_hard, F_hard_no_last, EARLY, PREFIX, AVG_LEN, STEPS, Y = run_dataset(jsonl)
            EARLY_clean = [e if e is not None else (max([x for x in EARLY if x is not None])+1) 
                for e in EARLY]
            results[model_name][L][lam] = {
                "corr_full"       : float(pearsonr(F_full, Y)[0]),
                "corr_hard"       : float(pearsonr(F_hard, Y)[0]),
                "corr_avg_prefix" : float(pearsonr(PREFIX, Y)[0]),
                "corr_avg_steps"  : float(pearsonr(STEPS, Y)[0]),
                "corr_avg_first_error"      : float(pearsonr(EARLY_clean, Y)[0]),
                "avg_prefix"      : float(np.mean(PREFIX)),
                "avg_first_error" : float(np.mean([e for e in EARLY if e is not None])),
                "avg_steps"       : float(np.mean(STEPS)),
                "file_used"       : jsonl,  # ⭐ Save for traceability
                "Y"               : Y
            }

            # 实时写盘（避免中断损失结果）
            with open(SAVE_PATH, "w") as f: json.dump(results, f, indent=4)
            print(f"💾 Saved to → {SAVE_PATH}\n")


