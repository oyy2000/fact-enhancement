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
# 观察者模型 (Student)
MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct" 
SAMPLE_LIMIT = 200 # 按照要求取 200 条

# 数据路径
PATH_SAME_FAMILY_14B = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/no_vector/gsm8k_cot_zeroshot_unified/Qwen2.5-14B-Instruct_L1_BASELINE/Qwen__Qwen2.5-14B-Instruct/samples_gsm8k_cot_zeroshot_unified_2026-01-22T16-17-22.512044.jsonl"
PATH_SAME_FAMILY_32B = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/no_vector/gsm8k_cot_zeroshot_unified/Qwen2.5-32B-Instruct_L1_BASELINE/Qwen__Qwen2.5-32B-Instruct/samples_gsm8k_cot_zeroshot_unified_2026-01-25T03-44-06.551722.jsonl"
PATH_SAME_FAMILY_72B = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/no_vector/gsm8k_cot_zeroshot_unified/Qwen2.5-72B-Instruct_L1_BASELINE/Qwen__Qwen2.5-72B-Instruct/samples_gsm8k_cot_zeroshot_unified_2026-01-22T21-14-55.626811.jsonl"
PATH_SAME_FAMILY_7B = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/no_vector/gsm8k_cot_zeroshot_unified/Qwen2.5-7B-Instruct_no_vector/Qwen__Qwen2.5-7B-Instruct/samples_gsm8k_cot_zeroshot_unified_2026-01-21T11-34-14.746371.jsonl"
PATH_CROSS_FAMILY = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/no_vector/gsm8k_cot_zeroshot_unified/Llama-3.1-8B-Instruct_no_vector/meta-llama__Llama-3.1-8B-Instruct/samples_gsm8k_cot_zeroshot_unified_2026-01-25T15-06-51.949587.jsonl"
PATH_GPT_OLD = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/gpt_rewrites_unified_new/Qwen_Qwen2.5-1.5B-Instruct/rewritten_old.json"
# ==========================================
# 1. 工具函数：读取数据 (支持 JSONL 和 JSON)
# ==========================================
def load_samples(file_path, limit=None, key=None):
    print(f"Loading data from: {os.path.basename(file_path)}...")
    data_dict = {}
    
    try:
        items = []
        if file_path.endswith('.json'):
            # 处理标准 JSON 列表
            with open(file_path, 'r', encoding='utf-8') as f:
                full_data = json.load(f)
                # 如果有limit，尽量多读一些以保证能找到交集，这里暂存全部或limit
                items = full_data if limit is None else full_data[:limit]
        else:
            # 处理 JSONL
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if limit is not None and i >= limit:
                        break
                    items.append(json.loads(line))

        for item in items:
            # 获取 doc_id
            if 'doc_id' not in item:
                continue
            
            doc_id = item['doc_id']
            
            # 获取 Question
            question = None
            # Check for doc['question'] (standard in lm_eval output)
            if "doc" in item and isinstance(item["doc"], dict) and "question" in item["doc"]:
                question = item["doc"]["question"]
            # Fallback for other formats
            elif "question" in item:
                question = item["question"]
            
            if not question:
                continue

            # 获取 Response
            text = None
            
            if key:
                # 指定 key
                if key in item:
                    text = item[key]
            else:
                # 默认逻辑: resps[0][0]
                if 'resps' in item and len(item['resps']) > 0:
                    text = item['resps'][0][0]
            
            if text:
                # 拼接 Question 和 Response
                prompt = qwen_chat_prompt(question)
                full_text = prompt + text
                data_dict[doc_id] = (prompt, full_text)
                    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return {}
    
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
# 3. 准备数据 & 4. 计算 NLL
# ==========================================
# 定义所有对比配置
configs = [
    {"name": "Same Family (Qwen-72B)", "path": PATH_SAME_FAMILY_72B, "key": None, "color": "#d62728"},
    {"name": "Same Family (Qwen-32B)", "path": PATH_SAME_FAMILY_32B, "key": None, "color": "#ff7f0e"},
    {"name": "Same Family (Qwen-14B)", "path": PATH_SAME_FAMILY_14B, "key": None, "color": "#9467bd"},
    
    {"name": "Same Family (Qwen-7B)", "path": PATH_SAME_FAMILY_7B, "key": None, "color": "#1f77b4"},
    {"name": "Cross Family (Llama-8B)", "path": PATH_CROSS_FAMILY, "key": None, "color": "yellow"},
    {"name": "GPT (Self / Original)", "path": PATH_GPT_OLD, "key": "resp_before", "color": "#17becf"},
    {"name": "GPT (Old)", "path": PATH_GPT_OLD, "key": "resp_after", "color": "#2ca02c"},
]

# 1. 加载所有数据到字典
print("\n--- Loading Data ---")
loaded_data = [] # Stores dicts {doc_id: text}
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

# 2. 找到共同 ID (Intersection of all loaded keys)
common_ids = set(loaded_data[0].keys())
for d in loaded_data[1:]:
    common_ids &= set(d.keys())

common_ids = sorted(list(common_ids))
print(f"Found {len(common_ids)} common samples across all {len(loaded_data)} datasets.")

if not common_ids:
    raise ValueError("No common samples found.")

# 截取
target_ids = common_ids[:SAMPLE_LIMIT]
print(f"Using {len(target_ids)} samples for evaluation.")

# 3. 计算 NLL 并存储结果
def calculate_nll(item_list, model, tokenizer):
    nlls = []
    # item_list is a list of tuples: (prompt, full_text)
    for prompt, full_text in tqdm(item_list, desc="Calculating NLL", leave=False):
        # 截断过长的文本
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        
        # 构建 Labels，默认全是 -100
        labels = inputs["input_ids"].clone()
        
        # 计算 prompt 的长度，以便进行 masking
        # 注意: 这里重新 encode prompt 可能会和 full text 的前缀有些微差异（例如空格处理），
        # 但通常对于 Chat 模板是准确的。
        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).input_ids
        prompt_len = prompt_ids.shape[1]
        
        # 只要 prompt 长度小于总长度，就进行 mask
        if prompt_len < labels.shape[1]:
            labels[:, :prompt_len] = -100
        else:
            # 如果 prompt 占满了或者比 full_text 还长（被截断），则 loss 无意义
            labels[:, :] = -100
            
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            
        # 如果全部被 mask，loss 可能是 nan，这里处理一下
        if torch.isnan(outputs.loss) or labels[0][0] == -100 and torch.all(labels == -100):
            nlls.append(0.0) # 或者 float('inf')
        else:
            nlls.append(outputs.loss.item())
    return nlls

# Define Cache File
CACHE_FILE = "nll_results_cache_v3_{}.json".format(MODEL_PATH.replace("/", "_"))
nll_cache = {}
if os.path.exists(CACHE_FILE):
    print(f"Loading results cache from {CACHE_FILE}...")
    with open(CACHE_FILE, 'r') as f:
        nll_cache = json.load(f)

results = []
print("\n--- Calculating NLL (with caching) ---")
for i, cfg in enumerate(valid_configs):
    name = cfg['name']
    print(f"[{i+1}/{len(valid_configs)}] Processing {name}...")
    
    if name not in nll_cache:
        nll_cache[name] = {}
        
    # Identify what needs to be calculated
    missing_ids = [doc_id for doc_id in target_ids if str(doc_id) not in nll_cache[name]]
    
    if missing_ids:
        print(f"  > Calculating missing NLL for {len(missing_ids)} items...")
        texts_to_run = [loaded_data[i][doc_id] for doc_id in missing_ids]
        scores = calculate_nll(texts_to_run, model, tokenizer)
        
        # Update Cache
        for doc_id, score in zip(missing_ids, scores):
            nll_cache[name][str(doc_id)] = score
        
        # Save cache immediately
        with open(CACHE_FILE, 'w') as f:
            json.dump(nll_cache, f)
    else:
        print(f"  > All {len(target_ids)} items found in cache.")
        
    # Retrieve aligned scores
    aligned_scores = [nll_cache[name][str(doc_id)] for doc_id in target_ids]
    results.append({
        "config": cfg,
        "scores": aligned_scores
    })

# 5. 统计与可视化
# ==========================================
print("\nGenerating visualization...")
plt.figure(figsize=(12, 8), dpi=150)
sns.set_style("whitegrid")

# 确定 X 轴范围
all_scores = [score for r in results for score in r["scores"]]
if not all_scores:
    x_min, x_max = 0, 1
else:
    x_min, x_max = min(all_scores) - 0.5, max(all_scores) + 0.5
x = np.linspace(x_min, 1, 1000)

print(f"\nResults Summary (Observer: Qwen-3B):")
for res in results:
    cfg = res["config"]
    scores = res["scores"]
    
    if not scores:
        continue
        
    mu, std = norm.fit(scores)
    print(f"{cfg['name']:<25}: Mean NLL = {mu:.4f}, Std = {std:.4f}")
    
    # Plot - Histogram (Faint, no label)
    plt.hist(scores, density=True, bins=30, alpha=0.1, color=cfg['color'])
    
    # Plot - Fit Curve (Bold)
    plt.plot(x, norm.pdf(x, mu, std), color=cfg['color'], linewidth=2.5, 
             label=f"{cfg['name']}\n$\mu={mu:.3f}, \sigma={std:.3f}$")

    # Plot - Mean Line (Vertical Dashed)
    plt.axvline(x=mu, color=cfg['color'], linestyle='--', linewidth=1.5, alpha=0.8)

plt.title("Distribution of Negative Log-Likelihood (NLL) on Model Responses", fontsize=16)
plt.xlabel("NLL (Lower means more likely/natural to Observer)", fontsize=14)
plt.ylabel("Probability Density", fontsize=14)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=10, borderaxespad=0.)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

save_path = f"distribution_alignment_{MODEL_PATH.replace('/', '_')}_all_variants.png"
plt.savefig(save_path)
print(f"\nVisualization saved to: {save_path}")