# scripts/01_extract_vectors.py
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm  # ✅ 进度条

TQDM_DISABLE = False if sys.stdout.isatty() else False  # 需要的话改成 True 关闭进度条

# -------- Config & Model --------
cfg = yaml.safe_load(open("config.yaml"))

tok = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
# 有些模型没有 pad token，补上以避免警告
if tok.pad_token_id is None and tok.eos_token_id is not None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    cfg["model_name"],
    torch_dtype=torch.float16,
    device_map="auto",               # 允许加速器做自动分片
)
model.eval()

# 读取数据与超参
# use one pairs 
pairs = [json.loads(l) for l in open("data/pairs.jsonl", "r")]
pairs = pairs[: cfg["num_pairs"]] if cfg["num_pairs"] > 0 else pairs
layers = list(cfg["layers_to_probe"])
span = int(cfg["token_span_first_step"])

num_layers = len(layers)
num_pairs = len(pairs)
total_fwds = num_layers * num_pairs * 2  # 每个样本做 true/false 两次前向

# -------- Helpers --------
@torch.inference_mode()
def get_hidden(prompt: str) -> torch.Tensor:
    """
    返回形状为 [n_layers, 1, T, d] 的 hidden states（去掉embedding层）。
    在 device_map='auto' 场景下，把 input_ids 放到 embedding 所在 device 上即可。
    """
    # 先在 CPU tokenize
    enc = tok(prompt, return_tensors="pt")
    # 把 input_ids 放到 embedding 的 device（分片/自动装载时更稳妥）
    embed_device = model.get_input_embeddings().weight.device
    input_ids = enc["input_ids"].to(embed_device)

    out = model(
        input_ids=input_ids,
        output_hidden_states=True,
        return_dict=True,
    )
    # out.hidden_states: tuple 长度 n_layers+1（含 embedding 层）
    hs = torch.stack(out.hidden_states, dim=0)  # [n_layers+1, 1, T, d]
    return hs[1:]  # 去掉 embedding 层 -> [n_layers, 1, T, d]


def first_step_span_ids(q: str, step1: str):
    """
    构造 prompt，并返回切片选取最后 span 个 token。
    """
    prompt = f"Q: {q}\nA: Let's think step by step. {step1}"
    ids = tok(prompt, return_tensors="pt")["input_ids"][0]
    T = ids.shape[0]
    sl = slice(max(0, T - span), T)
    return prompt, sl


# -------- Main Computation --------
W = {}

# 顶层总进度：统计所有前向调用（便于估算总 ETA）
overall = tqdm(total=total_fwds, desc="Total forward passes", disable=TQDM_DISABLE)

# 每层一个进度条
for L in tqdm(layers, desc="Layers", disable=TQDM_DISABLE):
    acc = []
    # 当前层的样本进度
    for ex in tqdm(pairs, desc=f"Layer {L} | examples", leave=False, disable=TQDM_DISABLE):
        p_true, sl_true = first_step_span_ids(ex["question"], ex["cot_step1_true"])
        p_false, sl_false = first_step_span_ids(ex["question"], ex["cot_step1_false"])

        Ht = get_hidden(p_true)[L, 0, sl_true, :]   # [span, d]
        overall.update(1)  # 记录一次前向

        Hf = get_hidden(p_false)[L, 0, sl_false, :] # [span, d]
        overall.update(1)  # 再记录一次前向

        diff = (Ht.mean(0) - Hf.mean(0)).float().cpu().numpy()  # [d]
        acc.append(diff)

    W[str(L)] = np.mean(np.stack(acc, axis=0), axis=0).tolist()

overall.close()

# -------- Save --------
Path("artifacts").mkdir(exist_ok=True, parents=True)
with open(f"artifacts/{cfg['model_name'].replace('/', '_')}_factual_dirs.json", "w") as f:
    json.dump(W, f)
print(f"Saved -> artifacts/{cfg['model_name'].replace('/', '_')}_factual_dirs.json")