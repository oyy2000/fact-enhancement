import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from tqdm import tqdm
import json
import os
import sys

# Add project root to sys.path to allow importing from utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils import qwen_chat_prompt

# ==========================================
# 配置参数
# ==========================================
MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct"  # Observer/Student
SAMPLE_LIMIT = 200

PATH_QWEN_14B = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/no_vector/gsm8k_cot_zeroshot_unified/Qwen2.5-14B-Instruct_no_vector/Qwen__Qwen2.5-14B-Instruct/samples_gsm8k_cot_zeroshot_unified_2026-01-22T16-17-22.512044.jsonl"
PATH_QWEN_32B = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/no_vector/gsm8k_cot_zeroshot_unified/Qwen2.5-32B-Instruct_no_vector/Qwen__Qwen2.5-32B-Instruct/samples_gsm8k_cot_zeroshot_unified_2026-01-25T03-44-06.551722.jsonl"
PATH_QWEN_72B = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/no_vector/gsm8k_cot_zeroshot_unified/Qwen2.5-72B-Instruct_no_vector/Qwen__Qwen2.5-72B-Instruct/samples_gsm8k_cot_zeroshot_unified_2026-01-22T21-14-55.626811.jsonl"
PATH_QWEN_7B  = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/no_vector/gsm8k_cot_zeroshot_unified/Qwen2.5-7B-Instruct_no_vector/Qwen__Qwen2.5-7B-Instruct/samples_gsm8k_cot_zeroshot_unified_2026-01-21T11-34-14.746371.jsonl"
PATH_LLAMA_1B = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/no_vector/gsm8k_cot_zeroshot_unified/Llama-3.2-1B-Instruct_no_vector/meta-llama__Llama-3.2-1B-Instruct/samples_gsm8k_cot_zeroshot_unified_2026-01-21T11-44-34.344707.jsonl"
PATH_LLAMA_3B = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/no_vector/gsm8k_cot_zeroshot_unified/Llama-3.2-3B-Instruct_no_vector/meta-llama__Llama-3.2-3B-Instruct/samples_gsm8k_cot_zeroshot_unified_2026-01-21T11-56-55.533987.jsonl"
PATH_LLAMA_8B = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/no_vector/gsm8k_cot_zeroshot_unified/Llama-3.1-8B-Instruct_no_vector/meta-llama__Llama-3.1-8B-Instruct/samples_gsm8k_cot_zeroshot_unified_2026-01-25T15-06-51.949587.jsonl"
PATH_LLAMA_70B = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/no_vector/gsm8k_cot_zeroshot_unified/Llama-3.1-70B-Instruct_no_vector/meta-llama__Llama-3.1-70B-Instruct/samples_gsm8k_cot_zeroshot_unified_2026-01-25T07-01-18.149083.jsonl"
PATH_GPT_REWRITTEN   = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/gpt_rewrites_unified_new/Qwen_Qwen2.5-3B-Instruct/rewritten_old.json"
PATH_GPT_5_1_PATH     = "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_gpt5_gsm8k_20260104_133223/gpt-5.1/samples_gsm8k_cot_zeroshot_2026-01-04T14-13-04.569148.jsonl"

# ==========================================
# 1. 工具函数：读取数据 (支持 JSONL 和 JSON)
# ==========================================
def load_samples(file_path, limit=None, key=None):
    print(f"Loading data from: {os.path.basename(file_path)}...")
    data_dict = {}

    items = []
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return {}

    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            full_data = json.load(f)
            items = full_data if limit is None else full_data[:limit]
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
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

        # response text
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

    print(f"Loaded {len(data_dict)} samples.")
    return data_dict

# ==========================================
# 2. 加载模型 (Observer)
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Observer Model: {MODEL_PATH} on {device}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()

# ==========================================
# 3. 配置对比组
# ==========================================
configs = [
    {"name": "Same Family (Qwen-72B)", "path": PATH_QWEN_72B, "key": None, "color": "#d62728"},
    {"name": "Same Family (Qwen-32B)", "path": PATH_QWEN_32B, "key": None, "color": "#ff7f0e"},
    {"name": "Same Family (Qwen-14B)", "path": PATH_QWEN_14B, "key": None, "color": "#9467bd"},
    {"name": "Same Family (Qwen-7B)",  "path": PATH_QWEN_7B,  "key": None, "color": "#1f77b4"},
    {"name": "Cross Family (Llama-8B)","path": PATH_LLAMA_8B,    "key": None, "color": "#bcbd22"},
    {"name": "Cross Family (Llama-70B)","path": PATH_LLAMA_70B,      "key": None, "color": "#8c564b"},
    {"name": "Cross Family (GPT-5.1)","path": PATH_GPT_5_1_PATH,   "key": None,  "color": "#e377c2"},
    {"name": "Self",                   "path": PATH_GPT_REWRITTEN,         "key": "resp_before", "color": "#17becf"},
    {"name": "Dense-Rewriting","path": PATH_GPT_REWRITTEN,         "key": "resp_after",  "color": "#2ca02c"},
]

# ==========================================
# 3.1 选择要绘制的曲线（白名单）
# ==========================================
SELECT_NAMES = [
    # "Same Family (Qwen-7B)",
    "Same Family (Qwen-14B)",
    # "Same Family (Qwen-32B)",
    # "Same Family (Qwen-72B)",
    # "Cross Family (Llama-8B)",
    # "Cross Family (Llama-70B)",
    "Dense-Rewriting",
    "Self",
    # "Cross Family (GPT-5.1)",
    # "Rewritten",
]

configs = [c for c in configs if c["name"] in set(SELECT_NAMES)]
print("Selected configs:", [c["name"] for c in configs])

print("\n--- Loading Data ---")
loaded_data = []
valid_configs = []

for cfg in configs:
    d = load_samples(cfg["path"], limit=None, key=cfg["key"])
    if d:
        loaded_data.append(d)
        valid_configs.append(cfg)
    else:
        print(f"Warning: Failed to load data for {cfg['name']}")

if not loaded_data:
    raise ValueError("No data loaded!")

# 共同 doc_id
common_ids = set(loaded_data[0].keys())
for d in loaded_data[1:]:
    common_ids &= set(d.keys())
common_ids = sorted(list(common_ids))

print(f"Found {len(common_ids)} common samples across all {len(loaded_data)} datasets.")
if not common_ids:
    raise ValueError("No common samples found.")

target_ids = common_ids[:SAMPLE_LIMIT]
print(f"Using {len(target_ids)} samples for evaluation.")

# ==========================================
# 4. 计算 NLL（更稳的mask + 丢弃无效样本）
# ==========================================
def nll_one(prompt: str, full_text: str, model, tokenizer, max_length: int = 2048):
    # tokenize full
    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(model.device)

    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    # prompt length: avoid re-adding special tokens to reduce mismatch risk
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length
    )["input_ids"]
    prompt_len = int(prompt_ids.shape[1])
    prompt_len = min(prompt_len, seq_len)  # clamp

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    # if everything masked -> invalid
    if torch.all(labels == -100):
        return None

    with torch.no_grad():
        outputs = model(**inputs, labels=labels)

    loss = outputs.loss
    if loss is None or torch.isnan(loss) or torch.isinf(loss):
        return None

    return float(loss.item())

# cache
CACHE_FILE = f"nll_results_cache_v5_{MODEL_PATH.replace('/', '_')}.json"
nll_cache = {}
if os.path.exists(CACHE_FILE):
    print(f"Loading results cache from {CACHE_FILE}...")
    with open(CACHE_FILE, "r") as f:
        nll_cache = json.load(f)

print("\n--- Calculating NLL (with caching) ---")

# 先逐 config 计算 raw nll（可能有 None）
raw_scores_by_cfg = {}
invalid_ids_global = set()

for idx_cfg, cfg in enumerate(valid_configs):
    name = cfg["name"]
    print(f"[{idx_cfg+1}/{len(valid_configs)}] Processing {name}...")

    if name not in nll_cache:
        nll_cache[name] = {}

    # 需要算的 doc
    missing_ids = [doc_id for doc_id in target_ids if str(doc_id) not in nll_cache[name]]
    if missing_ids:
        print(f"  > Calculating missing NLL for {len(missing_ids)} items...")
        for doc_id in tqdm(missing_ids, desc=f"NLL {name}", leave=False):
            prompt, full_text = loaded_data[idx_cfg][doc_id]
            score = nll_one(prompt, full_text, model, tokenizer)
            # score None 表示无效：暂存为 "__INVALID__"
            nll_cache[name][str(doc_id)] = "__INVALID__" if score is None else score

        with open(CACHE_FILE, "w") as f:
            json.dump(nll_cache, f)
    else:
        print(f"  > All {len(target_ids)} items found in cache.")

    # 读出 raw 分数，并记录无效 doc_id
    raw = {}
    for doc_id in target_ids:
        v = nll_cache[name].get(str(doc_id), "__INVALID__")
        if v == "__INVALID__":
            raw[doc_id] = None
            invalid_ids_global.add(doc_id)
        else:
            raw[doc_id] = float(v)
    raw_scores_by_cfg[name] = raw

# 同步丢弃：任何 config 无效的 doc_id，都从所有 config 移除，保证对齐公平
filtered_ids = [doc_id for doc_id in target_ids if doc_id not in invalid_ids_global]
print(f"\nFiltered invalid samples: {len(target_ids)} -> {len(filtered_ids)} (dropped {len(target_ids)-len(filtered_ids)})")

if len(filtered_ids) == 0:
    raise ValueError("All samples became invalid after masking/truncation. Increase max_length or check prompts.")

# 组装 results（对齐后的 list）
results = []
for cfg in valid_configs:
    name = cfg["name"]
    scores = [raw_scores_by_cfg[name][doc_id] for doc_id in filtered_ids]
    # 这里 scores 不该含 None 了
    results.append({"config": cfg, "scores": scores})


SELF_NAME = "Self"

# ==========================================
# 5. 可视化（论文风格）
# ==========================================
print("\nGenerating visualization (paper-style)...")

plt.figure(figsize=(10, 7), dpi=300)
sns.set_theme(
    style="whitegrid",
    context="paper",
    font_scale=1.2
)

# 统一 x 轴
all_scores = [s for r in results for s in r["scores"] if s is not None]
x_min = min(all_scores) - 0.4
x_max = max(all_scores) + 0.4
x = np.linspace(x_min, x_max, 1200)

print(f"\nResults Summary (Observer: {MODEL_PATH}):")


# ------------------------------------------
# Compute Self threshold (mean NLL)
# ------------------------------------------
self_mu = None
for res in results:
    if res["config"]["name"] == SELF_NAME:
        self_scores = np.array([s for s in res["scores"] if s is not None])
        self_mu = float(np.mean(self_scores))
        print(f"\n[Self baseline] Mean NLL = {self_mu:.4f}")
        break

assert self_mu is not None, "Self config not found!"

for res in results:
    cfg = res["config"]

    # Skip Self distribution (used as threshold only)
    if cfg["name"] == SELF_NAME:
        continue

    scores = np.array([s for s in res["scores"] if s is not None])

    if len(scores) < 5:
        print(f"{cfg['name']:<30}: Not enough valid samples ({len(scores)})")
        continue

    mu, std = norm.fit(scores)
    print(f"{cfg['name']:<30}: Mean NLL = {mu:.4f}, Std = {std:.4f}")

    # 1️⃣ 背景直方图（弱化）
    plt.hist(
        scores,
        bins=28,
        density=True,
        alpha=0.08,
        color=cfg["color"],
        edgecolor="none"
    )

    # 2️⃣ 高斯拟合曲线（主视觉）
    pdf = norm.pdf(x, mu, std)
    plt.plot(
        x, pdf,
        color=cfg["color"],
        linewidth=2.8,
        label=cfg["name"]
    )

    # 3️⃣ 均值中轴线
    plt.axvline(
        mu,
        color=cfg["color"],
        linestyle="--",
        linewidth=1.6,
        alpha=0.9
    )

    # 4️⃣ 在“中轴线”上直接标 μ
    y_mu = norm.pdf(mu, mu, std)
    plt.text(
        mu,
        y_mu * 1.03,   # 稍微抬高，避免压线
        f"$\\mu={mu:.2f}$",
        color=cfg["color"],
        fontsize=20,
        ha="center",
        va="bottom",
        rotation=90,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.75,
            pad=1.5
        )
    )


# ------------------------------------------
# Draw Self threshold line
# ------------------------------------------


y_top = plt.ylim()[1]

plt.axvline(
    self_mu,
    color="black",
    linestyle=":",
    linewidth=2.8,
    alpha=0.95,
    # label="Self baseline"
)
x_shift = -0.15

plt.text(
    self_mu + x_shift,
    y_top * 0.93,
    f"Self Baseline\n$\\mu={self_mu:.2f}$",
    ha="center",
    va="top",
    fontsize=20,
    color="black",
    bbox=dict(
        facecolor="white",
        edgecolor="black",
        alpha=0.85,
        pad=2
    )
)



# 坐标轴与整体样式
plt.xlabel("Negative Log-Likelihood (NLL)", fontsize=20, labelpad=8)
plt.ylabel("Probability Density", fontsize=20, labelpad=8)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.margins(x=0.02)
plt.xlim(x_min, x_max)

plt.legend(
    loc="upper right",      # 图内右上角
    frameon=True,
    fontsize=15,
    # title_fontsize=,
    borderpad=0.8,
    labelspacing=0.6
)


plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()

save_path = f"distribution_alignment_{MODEL_PATH.replace('/', '_')}_paper.png"
plt.savefig(save_path, bbox_inches="tight")
print(f"\nVisualization saved to: {save_path}")
