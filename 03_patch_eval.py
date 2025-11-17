# scripts/02_steer_eval.py
# -*- coding: utf-8 -*-
import json, re, random
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    set_seed,
)
import os

# -----------------------------
# Config, tokenizer, model
# -----------------------------
cfg = yaml.safe_load(open("config.yaml", "r"))

model_name = cfg["model_name"]
tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tok.pad_token_id is None and tok.eos_token_id is not None:
    tok.pad_token = tok.eos_token
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.get("cuda_visible_devices", "0")
seed = int(cfg.get("seed", 1234))
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); set_seed(seed)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
)
model.eval()

hidden_size = int(getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", 0))
if hidden_size <= 0:
    raise ValueError("Could not infer hidden_size from model.config")

# Steering directions: { "<layer_id>": [hidden_size floats], ... }
W = json.load(open("artifacts/factual_dirs.json", "r"))

max_new_tokens     = int(cfg.get("max_new_tokens", 128))
span               = int(cfg.get("token_span_first_step", 1))
steer_strengths    = list(cfg.get("steer_strengths", [0.0, 0.5, 1.0]))
truthfulqa_split   = str(cfg.get("truthfulqa_split", "validation"))
nli_model_name     = str(cfg.get("nli_model_name", "roberta-large-mnli"))
max_examples       = int(cfg.get("max_examples", 0))   # 0 -> use all

# -----------------------------
# Robust layer/MLP accessor
# -----------------------------
def get_block_and_mlp(model, layer_id: int):
    # LLaMA/Mistral
    p = getattr(model, "model", None)
    if p is not None and hasattr(p, "layers"):
        block = p.layers[layer_id]
        mlp = getattr(block, "mlp", None)
        if mlp is not None:
            return block, mlp
    # GPT-NeoX
    p = getattr(model, "gpt_neox", None)
    if p is not None and hasattr(p, "layers"):
        block = p.layers[layer_id]
        mlp = getattr(block, "mlp", None)
        if mlp is not None:
            return block, mlp
    # GPT-2 family
    p = getattr(model, "transformer", None)
    if p is not None and hasattr(p, "h"):
        block = p.h[layer_id]
        mlp = getattr(block, "mlp", None)
        if mlp is not None:
            return block, mlp

    raise AttributeError("Could not locate MLP module for this model family.")

# -----------------------------
# Generation with steering
# -----------------------------
def gen_with_steer(q: str, w_vec, layer_id: int, lam: float) -> str:
    w_np = np.array(w_vec, dtype=np.float32).reshape(-1)
    if w_np.shape[0] != hidden_size:
        raise ValueError(
            f"Direction length {w_np.shape[0]} != model hidden_size {hidden_size} (layer {layer_id})"
        )
    w = torch.tensor(w_np, dtype=model.dtype, device=model.device)

    _, mlp = get_block_and_mlp(model, layer_id)

    def hook_fn(module, inputs, output):
        out = output
        if isinstance(out, tuple):
            y = out[0]
            if not torch.is_tensor(y):  # unexpected structure
                return output
            y = y.clone()
            y[:, -span:, :] += lam * w
            return (y,) + out[1:]
        if torch.is_tensor(out):
            y = out.clone()
            y[:, -span:, :] += lam * w
            return y
        return output

    handle = mlp.register_forward_hook(hook_fn)

    # Keep a neutral, CoT-friendly prefix without external "first_step" (TruthfulQA has no GT CoT)
    prompt = f"{q}\nLet's think step by step."
    enc = tok(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}

    with torch.no_grad():
        gen_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

    handle.remove()
    return tok.decode(gen_ids[0], skip_special_tokens=True)

# -----------------------------
# --- 10) TruthfulQA loader + simple scoring ---
# (Your requested replacement)
# -----------------------------
def load_truthfulqa(split="validation"):
    import datasets as ds
    # truthful_qa has multiple configs; we'll use 'generation' version
    dset = ds.load_dataset("truthful_qa", "generation", split=split)
    # fields include: 'question', 'best_answer', 'correct_answers', 'incorrect_answers'
    return dset

def _infer_entailment_index(id2label: dict) -> int:
    """
    Map label name to entailment index robustly across NLI models.
    """
    # Typical for MNLI heads: {0:'CONTRADICTION',1:'NEUTRAL',2:'ENTAILMENT'} (case-insensitive)
    norm = {k: v.lower() for k, v in id2label.items()}
    for idx, name in norm.items():
        if "entail" in name:
            return int(idx)
    # Fallback to 2 (roberta-large-mnli)
    return 2

@torch.no_grad()
def entailment_probability(nli_tok, nli_model, premise: str, hypothesis: str) -> float:
    inputs = nli_tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(nli_model.device) for k, v in inputs.items()}
    logits = nli_model(**inputs).logits  # [1, num_labels]
    # softmax -> prob entailment
    probs = logits.softmax(dim=-1)[0]
    ent_idx = _infer_entailment_index(nli_model.config.id2label)
    return float(probs[ent_idx].item())

def truthfulqa_score(nli_tok, nli_model, model_answers: List[str], gt_answers: List[str]):
    """Compute a mixed score:
      - Exact match rate (case-insensitive) against any correct answer
      - NLI entailment prob of model answer w.r.t. the best_answer (as hypothesis true given premise best_answer)
    Returns dict with 'exact_match' and 'entailment_mean'.
    """
    exact = 0
    entail_scores = []
    for pred, gt in zip(model_answers, gt_answers):
        if pred.strip().lower() == gt.strip().lower():
            exact += 1
        ent = entailment_probability(nli_tok, nli_model, premise=gt, hypothesis=pred)
        entail_scores.append(ent)
    n = max(1, len(model_answers))
    return {
        "exact_match": exact / n,
        "entailment_mean": (sum(entail_scores) / len(entail_scores)) if entail_scores else float("nan")
    }

print("TruthfulQA helpers ready.")

# -----------------------------
# Load TruthfulQA + NLI model
# -----------------------------
print(f"Loading TruthfulQA split='{truthfulqa_split}' ...")
truthful = load_truthfulqa(truthfulqa_split)
questions = list(truthful["question"])
gt_best   = list(truthful["best_answer"])

if max_examples and max_examples > 0:
    questions = questions[:max_examples]
    gt_best   = gt_best[:max_examples]

print(f"Loaded {len(questions)} TruthfulQA examples.")

print(f"Loading NLI model '{nli_model_name}' ...")
nli_tok = AutoTokenizer.from_pretrained(nli_model_name, use_fast=True)
# Keep classification head in fp32 for stability; device_map="auto" handles placement
nli_model = AutoModelForSequenceClassification.from_pretrained(
    nli_model_name,
    torch_dtype=torch.float32,
    device_map="auto",
)
nli_model.eval()

# -----------------------------
# Main steering + TruthfulQA scoring
# -----------------------------
results = []
for L_str, w in W.items():
    L = int(L_str)
    for lam in steer_strengths:
        preds = []
        for q in questions:
            out = gen_with_steer(q, w, L, float(lam))
            # Extract only the model's continuation after the question (optional: keep full text)
            preds.append(out)

        scores = truthfulqa_score(nli_tok, nli_model, preds, gt_best)
        row = {
            "layer": L,
            "lambda": float(lam),
            "exact_match": float(scores["exact_match"]),
            "entailment_mean": float(scores["entailment_mean"]),
            "num_examples": len(preds),
        }
        results.append(row)
        print(f"[L={L:02d} lam={lam:+.2f}] EM={row['exact_match']:.3f}  NLI-Entail={row['entailment_mean']:.3f}  n={row['num_examples']}")

Path("artifacts").mkdir(exist_ok=True)
with open("artifacts/steer_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved -> artifacts/steer_results.json")
