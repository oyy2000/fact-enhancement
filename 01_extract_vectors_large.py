import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from pathlib import Path
from utils import qwen_chat_prompt
import os
# ==========================================
# ===== 引入 steering_vectors 库 ===========
# ==========================================
from steering_vectors import train_steering_vector, SteeringVector

# =====================
# ===== CONFIG ========
# =====================
EXPERIMENT_MODE = "GPT_REWRITE" 
TARGET_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
REWRITE_MODEL = "meta-llama/Llama-3.2-1B-Instruct" # "Qwen/Qwen2.5-7B-Instruct" # 仅在此模式下生效

# 层索引配置 (3B range 37)
model_name_to_layer_index = {
    # "Qwen/Qwen2.5-3B-Instruct": [i for i in range(37)],
    # "Qwen/Qwen2.5-1.5B-Instruct": [i for i in range(29)],
    # "Qwen/Qwen2.5-0.5B-Instruct": [i for i in range(25)],
    "meta-llama/Llama-3.2-1B-Instruct": [i for i in range(17)],
    # "meta-llama/Llama-3.2-3B-Instruct": [i for i in range(29)],
}

# 通用配置
NUM_EXAMPLES = 50

if EXPERIMENT_MODE == "GPT_REWRITE":
    DIR_PATH = "./gpt_rewrites_unified_new"
    PROMPT_STYLE = "old"  # 核心变量：仅在此模式下生效
    
    # 构造路径
    REWRITEEN_SAMPLE_PATH = os.path.join(DIR_PATH, TARGET_MODEL.replace("/", "_"))
    
    model_name_to_sample_paths = {
        TARGET_MODEL: os.path.join(REWRITEEN_SAMPLE_PATH, f"rewritten_{PROMPT_STYLE}.json")
    }
    
    # 输出目录
    root_out_dir = Path(REWRITEEN_SAMPLE_PATH) / f"vectors_{NUM_EXAMPLES}_{PROMPT_STYLE}"

elif EXPERIMENT_MODE == "LARGE_MODEL":
    # === 逻辑 2: Large Model Rewrites (Qwen 0.5B) ===
    DIR_PATH = "./large_model_rewrites_unified_new"
    
    # 构造路径
    REWRITEEN_SAMPLE_PATH = os.path.join(DIR_PATH, TARGET_MODEL.replace("/", "_"))
    
    model_name_to_sample_paths = {
        TARGET_MODEL: os.path.join(REWRITEEN_SAMPLE_PATH, f"{REWRITE_MODEL.replace('/', '_')}_paired_responses.json"),
    }
    
   
    # 输出目录
    root_out_dir = Path(REWRITEEN_SAMPLE_PATH) / f"vectors_{NUM_EXAMPLES}_paired_{REWRITE_MODEL.replace('/', '_')}"

else:
    raise ValueError(f"Unknown EXPERIMENT_MODE: {EXPERIMENT_MODE}")

# 创建输出目录
root_out_dir.mkdir(exist_ok=True, parents=True)

# ==========================================
# 打印检查 (Optional)
# ==========================================
print(f"Current Mode: {EXPERIMENT_MODE}")
print(f"Target Model: {TARGET_MODEL}")
print(f"Sample Path:  {model_name_to_sample_paths[TARGET_MODEL]}")
print(f"Output Dir:   {root_out_dir}")

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
    model_tag += "_applied"
    model_out_dir = root_out_dir / model_tag
    model_out_dir.mkdir(exist_ok=True)

    # 1. 加载模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,      
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

        # Calculate and save norms
        norms = {}
        # Check if layer_activations acts as a dict (keys are layer indices)
        for layer_idx, vec in steering_vector.layer_activations.items():
            # vec is likely a tensor of shape [hidden_dim] or [1, hidden_dim]
            norm_val = vec.norm().item()
            norms[layer_idx] = norm_val
            
        norms_path = model_out_dir / "vector_norms.json"
        with open(norms_path, "w") as f:
            json.dump(norms, f, indent=2)
        print(f"  ✔ Saved vector norms to {norms_path}")


    del model
    torch.cuda.empty_cache()

print("\nAll tasks completed.")