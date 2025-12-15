# prm_eval_core.py
import json, torch
import numpy as np
import torch.nn.functional as F
from scipy.stats import pearsonr
from transformers import AutoTokenizer, AutoModel
import os
import random

# ============================================================
# 🔒 FIXED RANDOM SEED FOR FULL DETERMINISM
# ============================================================
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Make CuDNN deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Disable some HF randomness (tokenization dropout etc)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONHASHSEED"] = str(SEED)

PRM_MODEL="Qwen/Qwen2.5-Math-PRM-7B"

# Global PRM model load
tokenizer = AutoTokenizer.from_pretrained(PRM_MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(
    PRM_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
).eval()

STEP_TOKEN = tokenizer.encode("<extra_0>")[0]

def prm_step_scores(logits, input_ids):
    probs = F.softmax(logits, dim=-1)
    idx = (input_ids == STEP_TOKEN).nonzero(as_tuple=True)[1]
    return [
        probs[0, idx[i]:idx[i+1], 1].mean().item()
        for i in range(len(idx)-1)
    ]

def eval_cot_prm(system, query, steps):
    text = "<extra_0>".join(steps) + "<extra_0>"
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
        {"role": "assistant", "content": text}
    ]
    conv = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    ids  = tokenizer.encode(conv, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    return prm_step_scores(out[0], ids)

def analyze_step_errors(scores, thr=0.5):
    err_idx = next((i+1 for i,s in enumerate(scores) if s < thr), None)
    prefix_len = 0
    for s in scores:
        if s > thr: prefix_len += 1
        else: break
    return err_idx, prefix_len

def avg_step_length(steps):
    return np.mean([len(s.split()) for s in steps]) if len(steps)>0 else 0

def run_dataset(jsonl, thr=0.5, label="exact_match"):
    data = [json.loads(l) for l in open(jsonl)]

    F_full=[]; F_no_last=[]; F_hard=[]; F_hard_no_last=[];
    EARLY=[]; PREFIX=[]; AVG_LEN=[]; STEPS=[];
    Y=[]

    STEP_SCORES_ALL = []   # ⭐ new
    STEP_TEXTS_ALL = []    # ⭐ new

    for d in data:
        if d.get("filter") == "strict-match":
            continue

        cot = d["resps"][0][0].strip()

        steps = (
            [s.strip() for s in cot.split("\n") if s.strip()]
            if "\n" in cot
            else [s.strip() for s in cot.split(".") if s.strip()]
        )

        if len(steps) == 0:
            continue

        # ---- safe eval, skip failure samples ----
        try:
            scores = eval_cot_prm("", d["arguments"]["gen_args_0"]["arg_0"], steps)
        except Exception as e:
            print(f"[WARN] PRM failed for one sample → skipping. Error: {e}")
            continue

        if len(scores) == 0:
            continue

        # ---- store raw step info ----
        STEP_SCORES_ALL.append(scores)
        STEP_TEXTS_ALL.append(steps)

        # ---- compute metrics ----
        Fi_full = sum(scores)/len(scores)
        Fi_no_last = sum(scores[:-1])/len(scores[:-1]) if len(scores)>1 else Fi_full
        Fi_hard = sum(s>thr for s in scores)/len(scores)
        Fi_hard_no_last = sum(s>thr for s in scores[:-1])/(len(scores)-1) if len(scores)>1 else Fi_hard

        earliest_err, prefix_len = analyze_step_errors(scores, thr)
        avg_len = avg_step_length(steps)
        yi = int(d.get(label, 0))

        # ---- append results ----
        F_full.append(Fi_full)
        F_no_last.append(Fi_no_last)
        F_hard.append(Fi_hard)
        F_hard_no_last.append(Fi_hard_no_last)
        EARLY.append(earliest_err)
        PREFIX.append(prefix_len)
        AVG_LEN.append(avg_len)
        STEPS.append(len(steps))
        Y.append(yi)

    # ---- ensure all lengths equal ----
    L = len(F_full)
    assert all(len(x) == L for x in [
        F_no_last, F_hard, F_hard_no_last,
        EARLY, PREFIX, AVG_LEN, STEPS, Y,
        STEP_SCORES_ALL, STEP_TEXTS_ALL
    ]), "❌ Length mismatch detected! Some sample was inconsistently added."

    return (
        F_full,
        F_no_last,
        F_hard,
        F_hard_no_last,
        EARLY,
        PREFIX,
        AVG_LEN,
        STEPS,
        Y,
        STEP_SCORES_ALL,   # ⭐ added
        STEP_TEXTS_ALL     # ⭐ added
    )
