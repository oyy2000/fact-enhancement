#!/usr/bin/env python3
"""
Compute per-sample NLL for the three audit rewrite configs
(DenseSteer, E3 Rule-Based, E3.2 GPT-5-mini) using Qwen-3B as observer.

Reuses the NLL computation from Preliminary_Analysis_KL_Divergence_Distribution_Shift.py.

Usage:
    python new_scripts/nll_audit_configs.py --gpu 0
"""

import torch
import json
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
from utils import qwen_chat_prompt

# ==========================================
# Paths
# ==========================================
BASE = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement"
MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct"

CONFIGS = [
    {
        "name": "DenseSteer (resp_before)",
        "path": os.path.join(BASE, "exps/gpt_rewrites_unified_new/Qwen_Qwen2.5-3B-Instruct/rewritten_old_50.json"),
        "key": "resp_before",
    },
    {
        "name": "DenseSteer (resp_after)",
        "path": os.path.join(BASE, "exps/gpt_rewrites_unified_new/Qwen_Qwen2.5-3B-Instruct/rewritten_old_50.json"),
        "key": "resp_after",
    },
    {
        "name": "E3 Rule-Based (resp_after)",
        "path": os.path.join(BASE, "control_experiments/Qwen_Qwen2.5-3B-Instruct/rewritten_e3_rule_based.json"),
        "key": "resp_after",
    },
    {
        "name": "E3.2 GPT-5-mini (resp_after)",
        "path": os.path.join(BASE, "control_experiments/Qwen_Qwen2.5-3B-Instruct/rewritten_e3_2_gpt54mini.json"),
        "key": "resp_after",
    },
]

# ==========================================
# Data loading (same as original script)
# ==========================================
def load_samples(file_path, key=None):
    print(f"Loading: {os.path.basename(file_path)} (key={key})")
    data_dict = {}
    items = []
    if not os.path.exists(file_path):
        print(f"  File not found: {file_path}")
        return data_dict

    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            items = json.load(f)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                items.append(json.loads(line))

    for item in items:
        if "doc_id" not in item:
            continue
        doc_id = item["doc_id"]
        question = None
        if "doc" in item and isinstance(item["doc"], dict) and "question" in item["doc"]:
            question = item["doc"]["question"]
        elif "question" in item:
            question = item["question"]
        if not question:
            continue

        text = None
        if key:
            text = item.get(key, None)
        else:
            if "resps" in item and len(item["resps"]) > 0 and len(item["resps"][0]) > 0:
                text = item["resps"][0][0]
        if not text:
            continue

        prompt = qwen_chat_prompt(question)
        full_text = prompt + text
        data_dict[doc_id] = (prompt, full_text)

    print(f"  Loaded {len(data_dict)} samples.")
    return data_dict

# ==========================================
# NLL computation (same as original script)
# ==========================================
def nll_one(prompt, full_text, model, tokenizer, max_length=2048):
    inputs = tokenizer(
        full_text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(model.device)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    prompt_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False,
        truncation=True, max_length=max_length
    )["input_ids"]
    prompt_len = min(int(prompt_ids.shape[1]), seq_len)

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    if torch.all(labels == -100):
        return None

    try:
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None

    loss = outputs.loss
    if loss is None or torch.isnan(loss) or torch.isinf(loss):
        return None
    return float(loss.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--output", default=os.path.join(BASE, "new_exps/e3/nll_audit_configs.json"))
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print(f"Loading observer model: {MODEL_PATH} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", torch_dtype=torch.float16
    ).eval()

    # Load all configs
    loaded = []
    for cfg in CONFIGS:
        d = load_samples(cfg["path"], key=cfg["key"])
        if d:
            loaded.append((cfg, d))

    # Common doc_ids
    common_ids = set(loaded[0][1].keys())
    for _, d in loaded[1:]:
        common_ids &= set(d.keys())
    common_ids = sorted(common_ids)
    print(f"\nCommon doc_ids across all configs: {len(common_ids)}")

    # Compute NLL
    all_results = {}
    for cfg, data in loaded:
        name = cfg["name"]
        print(f"\nComputing NLL for: {name}")
        scores = {}
        for doc_id in tqdm(common_ids, desc=name):
            prompt, full_text = data[doc_id]
            nll = nll_one(prompt, full_text, model, tokenizer)
            if nll is not None:
                scores[doc_id] = nll
        vals = list(scores.values())
        print(f"  {name}: n={len(vals)}, mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")
        all_results[name] = {
            "scores": scores,
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "n": len(vals),
        }

    # Summary table
    print(f"\n{'='*60}")
    print(f"  NLL Summary (observer: {MODEL_PATH})")
    print(f"{'='*60}")
    print(f"  {'Config':<35s}  {'Mean':>8s}  {'Std':>8s}  {'N':>4s}")
    print(f"  {'-'*35}  {'-'*8}  {'-'*8}  {'-'*4}")
    for name, r in all_results.items():
        print(f"  {name:<35s}  {r['mean']:>8.4f}  {r['std']:>8.4f}  {r['n']:>4d}")
    print(f"{'='*60}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
