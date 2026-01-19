import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import numpy as np
from pathlib import Path
from utils import qwen_chat_prompt

# ==========================================
# ===== 引入 steering_vectors 库 ===========
# ==========================================
from steering_vectors import train_steering_vector, SteeringVector

# =====================
# ===== CONFIG ========
# =====================

model_name_to_layer_index = {
    "Qwen/Qwen2.5-7B-Instruct": [14, 24, 28],
    "Qwen/Qwen2.5-1.5B-Instruct": [14, 24, 28],
    "Qwen/Qwen2.5-0.5B-Instruct": [12, 20, 24],
    "Qwen/Qwen2.5-3B-Instruct": [18, 32, 36],
}
MAX_EXAMPLES = 50

# 输出目录改为保存 SteeringVector 对象的目录
root_out_dir = Path(f"./vectors_Qwen14b_big_minus_small_selected_sample_{MAX_EXAMPLES}_lib")
root_out_dir.mkdir(exist_ok=True)

GPT_5_MODEL_SAMPLES_PATH = "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_gpt5_gsm8k_20260104_133223/gpt-5.1/samples_gsm8k_cot_zeroshot_2026-01-04T14-13-04.569148.jsonl"
QWEN_14B_MODEL_SAMPLES_PATH = "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_qwen_family/gsm8k_cot_zeroshot/Qwen2.5-14B-Instruct_L1_BASELINE/Qwen__Qwen2.5-14B-Instruct/samples_gsm8k_cot_zeroshot_2026-01-11T20-35-52.632309.jsonl"

model_name_to_sample_paths = {
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "big": QWEN_14B_MODEL_SAMPLES_PATH,
        "small": "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_qwen_family/gsm8k_cot_zeroshot/Qwen2.5-0.5B-Instruct_L1_BASELINE/Qwen__Qwen2.5-0.5B-Instruct/samples_gsm8k_cot_zeroshot_2026-01-11T19-30-46.284994.jsonl",
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "big": QWEN_14B_MODEL_SAMPLES_PATH,
        "small": "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_qwen_family/gsm8k_cot_zeroshot/Qwen2.5-1.5B-Instruct_L1_BASELINE/Qwen__Qwen2.5-1.5B-Instruct/samples_gsm8k_cot_zeroshot_2026-01-11T17-44-40.377584.jsonl",
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "big": QWEN_14B_MODEL_SAMPLES_PATH,
        "small": "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_qwen_family/gsm8k_cot_zeroshot/Qwen2.5-3B-Instruct_L1_BASELINE/Qwen__Qwen2.5-3B-Instruct/samples_gsm8k_cot_zeroshot_2026-01-11T18-39-08.765074.jsonl",
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "big": QWEN_14B_MODEL_SAMPLES_PATH,
        "small": "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_qwen_family/gsm8k_cot_zeroshot/Qwen2.5-7B-Instruct_L1_BASELINE/Qwen__Qwen2.5-7B-Instruct/samples_gsm8k_cot_zeroshot_2026-01-11T17-55-30.126533.jsonl",
    },
}

def get_exact_match(ex: dict):
    # (保持原有的 helper function 不变)
    if "exact_match" in ex:
        try:
            return float(ex["exact_match"])
        except Exception:
            pass
    for k in ["metrics", "metric", "results", "filtered", "eval", "scores"]:
        if k in ex and isinstance(ex[k], dict) and "exact_match" in ex[k]:
            try:
                return float(ex[k]["exact_match"])
            except Exception:
                pass
    for container_key in ["metrics", "results", "filtered", "scores"]:
        if container_key in ex and isinstance(ex[container_key], dict):
            for kk, vv in ex[container_key].items():
                if isinstance(kk, str) and "exact_match" in kk:
                    try:
                        return float(vv)
                    except Exception:
                        continue
    return None

# =====================
# ===== MAIN LOOP =====
# =====================

for model_name, layer_list in model_name_to_layer_index.items():

    print(f"\n========== Processing model: {model_name} ==========")

    # ---- output dir per model ----
    model_tag = model_name.replace("/", "_")
    model_out_dir = root_out_dir / model_tag
    model_out_dir.mkdir(exist_ok=True)

    # =====================
    # ===== MODEL =========
    # =====================

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    print(model_name, "loaded")

    # =====================
    # ===== DATA ==========
    # =====================

    paths = model_name_to_sample_paths[model_name]
    with open(paths["big"], "r") as f:
        big_data = [json.loads(line) for line in f]

    with open(paths["small"], "r") as f:
        small_data = [json.loads(line) for line in f]

    big_by_id   = {int(x["doc_id"]): x for x in big_data}
    small_by_id = {int(x["doc_id"]): x for x in small_data}

    doc_ids = sorted(set(big_by_id) & set(small_by_id))

    # ===== 统计与筛选 =====
    TAIL_N = 200
    tail_ids = doc_ids[-TAIL_N:] if len(doc_ids) > TAIL_N else doc_ids

    cnt_1, cnt_0, cnt_missing = 0, 0, 0
    for did in tail_ids:
        em = get_exact_match(big_by_id[did])
        if em is None: cnt_missing += 1
        elif em >= 0.5: cnt_1 += 1
        else: cnt_0 += 1

    print(f"[Stats] last {len(tail_ids)} doc_ids: exact_match=1.0 -> {cnt_1}, 0.0 -> {cnt_0}, missing -> {cnt_missing}")

    # (B) 筛选样本
    selected_ids = []
    for did in tail_ids:
        em = get_exact_match(big_by_id[did])
        if em is not None and em >= 0.5:
            selected_ids.append(did)

    if MAX_EXAMPLES > 0:
        selected_ids = selected_ids[:MAX_EXAMPLES]

    print("steering doc_ids (tail exact_match==1.0):", len(selected_ids))

    # ==========================================
    # ===== 构建 Training Samples for Lib ======
    # ==========================================
    # 库通常期望输入是成对的文本 (positive_text, negative_text)
    # 这里的 "positive" 是 big 模型的回答（更好的回答），"negative" 是 small 模型的回答
    steering_data = []
    
    for did in selected_ids:
        ex_big = big_by_id[did]
        ex_small = small_by_id[did]
        q = ex_big["doc"]["question"]
        
        prompt = qwen_chat_prompt(q)
        
        resp_big = ex_big["resps"][0][0]
        resp_small = ex_small["resps"][0][0]
        
        text_pos = prompt + resp_big
        text_neg = prompt + resp_small
        
        steering_data.append((text_pos, text_neg))

    # ==========================================
    # ===== 使用 Library 提取 Vector ===========
    # ==========================================
    
    if len(steering_data) > 0:
        print(f"  → Extracting vectors for layers {layer_list} using steering-vectors lib...")
        
        # train_steering_vector 会自动处理 forward pass、last token extraction 和 aggregation (PCA/Mean)
        # 默认使用 PCA 提取主要方向，如果你想要简单的平均差，可以设置 aggregation="mean"
        steering_vector = train_steering_vector(
            model, 
            tokenizer,
            steering_data,
            layers=layer_list,
            move_to_cpu=True, # 提取完后移回 CPU 节省显存
            # aggregation="mean", # 如果你想要纯平均值，取消注释这行。默认是 "pca"
            # batch_size=4, # 如果显存允许，可以增大 batch_size 加速
        )

        # 保存为库的标准格式
        save_path = model_out_dir / "steering_vector.pt"
        steering_vector.save(save_path)
        print(f"  ✔ Saved steering vector object to: {save_path}")
        
    else:
        print("  ⚠ No samples selected, skipping extraction.")

    del model
    torch.cuda.empty_cache()

print("\nAll models processed.")