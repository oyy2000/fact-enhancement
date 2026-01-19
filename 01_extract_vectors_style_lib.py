import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils import qwen_chat_prompt

# ==========================================
# ===== 引入 steering_vectors 库 ===========
# ==========================================
from steering_vectors import train_steering_vector, SteeringVector

# =====================
# ===== CONFIG ========
# =====================

model_name_to_layer_index = {
    "Qwen/Qwen2.5-3B-Instruct": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
}
NUM_EXAMPLES = 50

# 输出目录_
root_out_dir = Path(f"./vectors_less_tokens_compare{NUM_EXAMPLES}_lib")
root_out_dir.mkdir(exist_ok=True)

QWEN_3B_MODEL_LESS_TOKENS_SAMPLES_PATH = "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_qwen_family/gsm8k_cot_zeroshot/Qwen2.5-3B-Instruct_L1_BASELINE/Qwen__Qwen2.5-3B-Instruct/samples_gsm8k_cot_zeroshot_rewritten_less_tokens_per_step.json"

model_name_to_sample_paths = {
    "Qwen/Qwen2.5-3B-Instruct": QWEN_3B_MODEL_LESS_TOKENS_SAMPLES_PATH
}

# =====================
# ===== HELPERS =======
# =====================

def get_exact_match(ex: dict):
    if "exact_match" in ex:
        try: return float(ex["exact_match"])
        except: pass
    for k in ["metrics", "results", "scores"]:
        if k in ex and isinstance(ex[k], dict) and "exact_match" in ex[k]:
            try: return float(ex[k]["exact_match"])
            except: pass
    return None

def load_samples(path: str):
    path = str(path)
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    else:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            for k in ["samples", "instances", "data"]:
                if k in obj and isinstance(obj[k], list): return obj[k]
        return obj

# =====================
# ===== MAIN LOOP =====
# =====================

for model_name, layer_list in model_name_to_layer_index.items():
    print(f"\n========== Processing model: {model_name} ==========")

    model_tag = model_name.replace("/", "_")
    model_out_dir = root_out_dir / model_tag
    model_out_dir.mkdir(exist_ok=True)

    # 1. 加载模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    # 2. 准备数据
    sample_path = model_name_to_sample_paths[model_name]
    samples = load_samples(sample_path)
    
    by_id = {ex["doc_id"]: ex for ex in samples if "doc_id" in ex}
    doc_ids = sorted(by_id.keys())
    
    # 筛选准确率 1.0 的样本（从后往前）
    selected_ids = []
    for did in reversed(doc_ids):
        if get_exact_match(by_id[did]) == 1.0:
            selected_ids.append(did)
            if len(selected_ids) >= NUM_EXAMPLES: break
    selected_ids = list(reversed(selected_ids))

    print(f"Selected {len(selected_ids)} samples for steering.")

    # 3. 构造正负样本对 (positive, negative)
    # 根据你的逻辑：rewritten 是更好的行为 (Positive)，original 是之前的行为 (Negative)
    training_samples = []
    for did in selected_ids:
        ex = by_id[did]
        prompt = qwen_chat_prompt(ex["doc"]["question"])
        training_samples.append((prompt + ex["resp_after"], prompt + ex["resp_before"]))

    # 4. 训练转向向量 (使用文档中的 train_steering_vector)
    if training_samples:
        print(f"  → Training steering vector for layers: {layer_list}")
        
        # 按照文档参数调用
        steering_vector = train_steering_vector(
            model=model,
            tokenizer=tokenizer,
            training_samples=training_samples,
            layers=layer_list,
            layer_type="decoder_block", # Qwen 默认使用 decoder 结构
            move_to_cpu=True,           # 节省显存，将结果存放在 CPU
            read_token_index=-1,        # 文档默认值，读取最后一个 token 的激活值
            show_progress=True,
            batch_size=1                # 如果显存充足可以调大
        )

        # 5. 保存结果
        # 注意：使用 torch.save 存储 SteeringVector 对象
        save_path = model_out_dir / "steering_vector.pt"
        torch.save(steering_vector, save_path)
        
        print(f"  ✔ Successfully saved SteeringVector to {save_path}")
        # steering_vector.layer_activations 是一个 dict {layer_idx: tensor}
        print(f"     Layers in object: {list(steering_vector.layer_activations.keys())}")

    del model
    torch.cuda.empty_cache()

print("\nAll tasks completed.")