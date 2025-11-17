# scripts/02_steer_eval_triviaqa.py
# -*- coding: utf-8 -*-
import json, random, re, string
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

import numpy as np
import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)

# =============================
# 抽答案 & 文本规范化 & 评分
# =============================
def extract_answer(text: str) -> str:
    s = text.strip()
    markers = [
        r"(?:^|\n)\s*final answer\s*:\s*(.*)",
        r"(?:^|\n)\s*answer\s*:\s*(.*)",
        r"(?:^|\n)\s*a\s*:\s*(.*)",
        r"(?:^|\n)\s*final\s*:\s*(.*)",
        r"(?:^|\n)\s*最终答案\s*[:：]\s*(.*)",
        r"(?:^|\n)\s*结论\s*[:：]\s*(.*)",
    ]
    for pat in markers:
        m = re.search(pat, s, flags=re.IGNORECASE | re.DOTALL)
        if m:
            cand = m.group(1).strip()
            cand = cand.splitlines()[0].strip()
            return _clean_inline(cand)
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if lines:
        return _clean_inline(lines[-1])
    sent = re.split(r"(?<=[。！？.!?])\s+", s)
    return _clean_inline(sent[0].strip()) if sent and sent[0].strip() else s

def _clean_inline(s: str) -> str:
    s = s.strip().strip('\"“”\'')
    s = re.sub(r"\s*(?:#|//).*?$", "", s).strip()
    return s

def normalize_answer(s: str) -> str:
    """SQuAD 风格：小写、去标点、去冠词、规范空白"""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 and len(truth_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0
    common = {}
    for tok in pred_tokens:
        common[tok] = min(pred_tokens.count(tok), truth_tokens.count(tok))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)

def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def metric_max_over_ground_truths(pred: str, gts: List[str]) -> Tuple[float, float]:
    """对多个参考答案取最大 EM / F1"""
    if not gts:
        return 0.0, 0.0
    em = max(exact_match_score(pred, gt) for gt in gts)
    f1 = max(f1_score(pred, gt) for gt in gts)
    return float(em), float(f1)

def write_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# =============================
# Config & Model
# =============================
cfg = yaml.safe_load(open("config.yaml", "r"))
model_name = cfg["model_name"]

tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tok.pad_token_id is None and tok.eos_token_id is not None:
    tok.pad_token = tok.eos_token

seed = int(cfg.get("seed", 1234))
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); set_seed(seed)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

hidden_size = int(getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", 0))
if hidden_size <= 0:
    raise ValueError("Could not infer hidden_size from model.config")

W = json.load(open("artifacts/factual_dirs.json", "r"))
max_new_tokens = int(cfg.get("max_new_tokens", 128))
span = int(cfg.get("token_span_first_step", 1))
steer_strengths = list(cfg.get("steer_strengths", [0.0, 0.5, 1.0]))

# TriviaQA 相关配置（可放到 config.yaml）
triviaqa_subset = str(cfg.get("triviaqa_subset", "rc"))           # "rc" 或 "unfiltered"
triviaqa_split  = str(cfg.get("triviaqa_split", "validation"))    # "train" / "validation"

max_examples = int(cfg.get("max_examples", 0))

# 输出路径
Path("artifacts").mkdir(exist_ok=True)
GEN_JSONL = "artifacts/triviaqa_outputs.jsonl"
RESULTS_JSON = "artifacts/steer_results_triviaqa.json"
open(GEN_JSONL, "w").close()  # 若想追加历史可注释

# =============================
# 模型结构（Hook层）
# =============================
def get_block_and_mlp(model, layer_id: int):
    p = getattr(model, "model", None)
    if p and hasattr(p, "layers"):
        block = p.layers[layer_id]
        return block, getattr(block, "mlp", None)
    p = getattr(model, "gpt_neox", None)
    if p and hasattr(p, "layers"):
        block = p.layers[layer_id]
        return block, getattr(block, "mlp", None)
    p = getattr(model, "transformer", None)
    if p and hasattr(p, "h"):
        block = p.h[layer_id]
        return block, getattr(block, "mlp", None)
    raise AttributeError("Could not locate MLP module for this model family.")

# =============================
# 只返回新生成文本
# =============================
def _generate_completion(prompt_ids, **gen_kwargs):
    input_len = prompt_ids["input_ids"].shape[1]
    gen_ids = model.generate(**prompt_ids, **gen_kwargs)
    new_ids = gen_ids[0, input_len:]
    return tok.decode(new_ids, skip_special_tokens=True)

def _make_prompt(q: str) -> str:
    # 也可以改成在末尾强制 "Final answer:" 提示，利于抽取
    return f"{q}\nLet's think step by step."

# =============================
# 带 Steering 的生成
# =============================
def gen_with_steer(q: str, w_vec, layer_id: int, lam: float) -> str:
    w_np = np.array(w_vec, dtype=np.float32).reshape(-1)
    if w_np.shape[0] != hidden_size:
        raise ValueError(
            f"Direction length {w_np.shape[0]} != hidden_size {hidden_size} (layer {layer_id})"
        )
    w = torch.tensor(w_np, dtype=model.dtype, device="cpu", requires_grad=False)

    _, mlp = get_block_and_mlp(model, layer_id)

    def hook_fn(module, inputs, output):
        out = output
        if isinstance(out, tuple):
            y = out[0]
            if not torch.is_tensor(y):
                return output
            y = y.clone()
            local_w = w.to(device=y.device, dtype=y.dtype)
            t = y.size(1)
            sl = min(span, t)
            y[:, t - sl :, :] += lam * local_w
            return (y,) + out[1:]
        elif torch.is_tensor(out):
            y = out.clone()
            local_w = w.to(device=y.device, dtype=y.dtype)
            t = y.size(1)
            sl = min(span, t)
            y[:, t - sl :, :] += lam * local_w
            return y
        return output

    handle = mlp.register_forward_hook(hook_fn)
    prompt = _make_prompt(q)
    enc = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)

    with torch.no_grad():
        gen_text = _generate_completion(
            enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

    handle.remove()
    return gen_text

# =============================
# 加载 TriviaQA
# =============================
def load_triviaqa(subset="rc", split="validation"):
    import datasets as ds
    ds_ = ds.load_dataset("trivia_qa", subset, split=split)
    # 统一取 question & 多参考答案列表
    questions, answers_multi = [], []
    for ex in ds_:
        q = ex.get("question", "").strip()
        ans = ex.get("answer", {})
        # 官方字段：answer.value（首选） 与 answer.aliases（同义表述）
        refs = []
        if isinstance(ans, dict):
            if "value" in ans and ans["value"]:
                refs.append(str(ans["value"]).strip())
            if "aliases" in ans and ans["aliases"]:
                refs.extend([str(a).strip() for a in ans["aliases"] if str(a).strip()])
        refs = list({r for r in refs if r})  # 去重
        if q and refs:
            questions.append(q)
            answers_multi.append(refs)
    return questions, answers_multi

print(f"Loading TriviaQA subset='{triviaqa_subset}' split='{triviaqa_split}' ...")
questions, refs_multi = load_triviaqa(triviaqa_subset, triviaqa_split)

# 可快速调试：限制样本数
if max_examples and max_examples > 0:
    questions = questions[:max_examples]
    refs_multi = refs_multi[:max_examples]
print(f"Loaded {len(questions)} examples.")

# =============================
# 主循环：Baseline + Steering
# =============================
results = []
print("\n[Baseline λ=0.00] Generating TriviaQA answers (no steering)...")
baseline_preds = []
for i, (q, gts) in enumerate(tqdm(list(zip(questions, refs_multi)), desc="Baseline", ncols=90)):
    prompt = _make_prompt(q)
    enc = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        gen_text = _generate_completion(
            enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    ans = extract_answer(gen_text)
    baseline_preds.append(ans)

    # 保存逐条
    write_jsonl(GEN_JSONL, {
        "dataset": "trivia_qa",
        "subset": triviaqa_subset,
        "split": triviaqa_split,
        "mode": "baseline",
        "layer": None,
        "lambda": 0.0,
        "index": i,
        "question": q,
        "gt_refs": gts,
        "generation_raw": gen_text,
        "answer_extracted": ans,
        "model_name": model_name,
    })

# 计算 Baseline EM/F1
em_sum, f1_sum = 0.0, 0.0
for pred, gts in zip(baseline_preds, refs_multi):
    em, f1 = metric_max_over_ground_truths(pred, gts)
    em_sum += em
    f1_sum += f1
n = max(1, len(baseline_preds))
baseline_scores = {"exact_match": em_sum / n, "f1": f1_sum / n}
print(f"Baseline → EM={baseline_scores['exact_match']:.3f} | F1={baseline_scores['f1']:.3f}")

results.append({
    "layer": None,
    "lambda": 0.0,
    "exact_match": float(baseline_scores["exact_match"]),
    "f1": float(baseline_scores["f1"]),
    "num_examples": len(baseline_preds),
})
with open(RESULTS_JSON, "w") as f:
    json.dump(results, f, indent=2)

# === Steering runs ===
for L_str, w in W.items():
    L = int(L_str)
    for lam in steer_strengths:
        preds = []
        print(f"\n[L={L:02d} λ={lam:+.2f}] Generating TriviaQA answers...")
        for i, (q, gts) in enumerate(tqdm(list(zip(questions, refs_multi)), desc=f"Layer {L}, λ={lam:+.2f}", ncols=90)):
            gen_text = gen_with_steer(q, w, L, float(lam))
            ans = extract_answer(gen_text)
            preds.append(ans)

            write_jsonl(GEN_JSONL, {
                "dataset": "trivia_qa",
                "subset": triviaqa_subset,
                "split": triviaqa_split,
                "mode": "steer",
                "layer": L,
                "lambda": float(lam),
                "index": i,
                "question": q,
                "gt_refs": gts,
                "generation_raw": gen_text,
                "answer_extracted": ans,
                "model_name": model_name,
            })

        # 计算 EM/F1
        em_sum, f1_sum = 0.0, 0.0
        for pred, gts in zip(preds, refs_multi):
            em, f1 = metric_max_over_ground_truths(pred, gts)
            em_sum += em
            f1_sum += f1
        em = em_sum / max(1, len(preds))
        f1 = f1_sum / max(1, len(preds))

        delta_em  = em - baseline_scores["exact_match"]
        delta_f1  = f1 - baseline_scores["f1"]

        row = {
            "layer": L,
            "lambda": float(lam),
            "exact_match": float(em),
            "f1": float(f1),
            "delta_em": float(delta_em),
            "delta_f1": float(delta_f1),
            "num_examples": len(preds),
        }
        results.append(row)
        print(f"  → EM={em:.3f} (Δ{delta_em:+.3f}) | F1={f1:.3f} (Δ{delta_f1:+.3f})")

        with open(RESULTS_JSON, "w") as f:
            json.dump(results, f, indent=2)

print(f"\nAll results saved -> {RESULTS_JSON} ✅")
print(f"Per-sample generations saved -> {GEN_JSONL} ✅")
