import json, torch
import numpy as np
import torch.nn.functional as F
from scipy.stats import pearsonr
from transformers import AutoTokenizer, AutoModel
import os
import matplotlib.pyplot as plt
PRM_MODEL="Qwen/Qwen2.5-Math-PRM-7B"


tokenizer=AutoTokenizer.from_pretrained(PRM_MODEL,trust_remote_code=True)
model=AutoModel.from_pretrained(
    PRM_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
).eval()

STEP_TOKEN = tokenizer.encode("<extra_0>")[0]

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
# ---------------- Main Evaluation ----------------
def run_dataset(jsonl, thr=0.5, label="exact_match"):
    data=[json.loads(l) for l in open(jsonl)]

    F_full=[]              # using all steps
    F_no_last=[]           # removing last step
    F_hard=[]              # hard-threshold
    F_hard_no_last=[]      # hard-threshold + no last step
    
    EARLY=[]               # earliest error step
    PREFIX=[]              # prefix correctness length
    Y=[]                   # final answer correctness

    for i,d in enumerate(data):
        if d.get("filter") == "strict-match":
            continue
        
        cot = d["resps"][0][0].strip()

        # split (fallback to '.' if no newline)
        steps = [s.strip() for s in cot.split("\n") if s.strip()] \
                if "\n" in cot else \
                [s.strip() for s in cot.split(".") if s.strip()]

        scores = eval_cot_prm("", d["arguments"]["gen_args_0"]["arg_0"], steps)

        # ===== metric A: mean of all steps =====
        Fi_full   = sum(scores)/len(scores) if scores else 0
        
        # ===== metric B: remove last reasoning step =====
        Fi_no_last = sum(scores[:-1])/len(scores[:-1]) if len(scores)>1 else Fi_full
        
        # ===== metric C: convert to 0/1 by threshold =====
        Fi_hard = sum([s>thr for s in scores]) / len(scores)

        # ===== metric D: convert to 0/1 by threshold + no last step =====
        Fi_hard_no_last = sum([s>thr for s in scores[:-1]]) / len(scores[:-1]) if len(scores)>1 else Fi_hard

        # ===== NEW: earliest error + prefix correctness =====
        earliest_err, prefix_len = analyze_step_errors(scores, thr)
        
        yi = int(d.get(label,0))

        print(f"[{i}] steps={len(steps)} scores={scores}")
        print(f"  Full={Fi_full:.3f}  NoLast={Fi_no_last:.3f} | Hard={Fi_hard:.3f} HardNoLast={Fi_hard_no_last:.3f}")
        print(f"  🔹EarliestError={earliest_err}   🔹PrefixCorrectLen={prefix_len}  | Y={yi}\n")

        F_full.append(Fi_full)
        F_no_last.append(Fi_no_last)
        F_hard.append(Fi_hard)
        F_hard_no_last.append(Fi_hard_no_last)
        EARLY.append(earliest_err)
        PREFIX.append(prefix_len)
        Y.append(yi)

    # ---- compute & print correlations ----
    def corr(x): return pearsonr(x,Y)[0], pearsonr(x,Y)[1]

    r1,p1=corr(F_full)
    r2,p2=corr(F_no_last)
    r3,p3=corr(F_hard)
    r4,p4=corr(F_hard_no_last)

    print("\n━━━━━━━━ Result Summary ━━━━━━━━")
    print(f"🔵 Full-step corr     = {r1:.4f} (p={p1:.2e})")
    print(f"🟠 Without-final-step = {r2:.4f} (p={p2:.2e})")
    print(f"🟥 Hard-label corr    = {r3:.4f} (p={p3:.2e})")
    print(f"🟣 Hard-label NoLast  = {r4:.4f} (p={p4:.2e})")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    return F_full, F_no_last, F_hard, F_hard_no_last, EARLY, PREFIX, Y




# ===================== CONFIG =====================
BASE_DIR = "/home/youyang7/projects/lm-evaluation-harness/lm_eval/models/eval_grid_11_18/triviaqa_cot_gsm8k_cot_zeroshot"

L_values   = ["L8", "L16", "L24"]
lam_values = ["BASELINE","lam0p5","lam1p0","lam1p5","lam2p0"]   # 横轴维度

# λ label 显示用
lam_plot_labels = {
    "BASELINE":"0.0",
    "lam0p5":"0.5",
    "lam1p0":"1.0",
    "lam1p5":"1.5",
    "lam2p0":"2.0"
}
# ==================================================

results = {}   # results[L][lam] = dict of metrics

for L in L_values:
    results[L] = {}
    for lam in lam_values:

        # ===== 生成 JSONL 路径 =====
        folder = f"Mistral-7B-Instruct-v0.3_{L}_{lam}"
        jsonl_path = f"{BASE_DIR}/{folder}/mistralai__Mistral-7B-Instruct-v0.3/" \
                     f"samples_gsm8k_cot_zeroshot_*.jsonl"

        # 递归查找文件（因为你文件名带时间戳）
        files = [os.path.join(dp,f) 
                 for dp,_,fs in os.walk(f"{BASE_DIR}/{folder}") 
                 for f in fs if f.endswith(".jsonl")]

        if len(files)==0:
            print(f"⚠ No file found for {L}-{lam}")
            continue

        jsonl = sorted(files)[-1]   # 选最新文件  ← 你时间戳命名恰好适配
        print(f"\n🚀 Running {L}-{lam}\n -> {jsonl}\n")

        # ===== 调用你的评估函数 =====
        F_full, F_no_last, F_hard, F_hard_no_last, EARLY, PREFIX, Y = run_dataset(jsonl)

        # 保存结果
        results[L][lam] = {
            "corr_full"       : pearsonr(F_full, Y)[0],
            "corr_no_last"    : pearsonr(F_no_last, Y)[0],
            "corr_hard"       : pearsonr(F_hard, Y)[0],
            "corr_hard_nl"    : pearsonr(F_hard_no_last, Y)[0],
            "avg_prefix"      : np.mean([p for p in PREFIX if p is not None]),
            "avg_first_error" : np.mean([e for e in EARLY if e is not None])
        }

