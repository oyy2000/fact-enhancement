import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from utils import qwen_chat_prompt
# =====================
# ===== CONFIG ========
# =====================

model_name_to_layer_index = {
    "Qwen/Qwen2.5-7B-Instruct": [14, 24, 28],
    "Qwen/Qwen2.5-1.5B-Instruct": [14, 24, 28],
    "Qwen/Qwen2.5-3B-Instruct": [18, 32, 36],
}
MAX_EXAMPLES = 10

root_out_dir = Path(f"./vectors_Qwen14b_big_minus_small_selected_sample_{MAX_EXAMPLES}")
root_out_dir.mkdir(exist_ok=True)

GPT_5_MODEL_SAMPLES_PATH = "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_gpt5_gsm8k_20260104_133223/gpt-5.1/samples_gsm8k_cot_zeroshot_2026-01-04T14-13-04.569148.jsonl"
QWEN_14B_MODEL_SAMPLES_PATH = "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_qwen_familiy/gsm8k_cot_zeroshot/Qwen2.5-14B-Instruct_L8_BASELINE/Qwen__Qwen2.5-14B-Instruct/samples_gsm8k_cot_zeroshot_2025-12-11T09-28-44.120949.jsonl"

model_name_to_sample_paths = {
    "Qwen/Qwen2.5-7B-Instruct": {
        "big": QWEN_14B_MODEL_SAMPLES_PATH,
        # "big": GPT_5_MODEL_SAMPLES_PATH,
        "small": "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_qwen_familiy/gsm8k_cot_zeroshot/Qwen2.5-7B-Instruct_L8_BASELINE/Qwen__Qwen2.5-7B-Instruct/samples_gsm8k_cot_zeroshot_2025-12-01T01-41-34.372070.jsonl",
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        # "big": "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_qwen_familiy/gsm8k_cot_zeroshot/Qwen2.5-7B-Instruct_L8_BASELINE/Qwen__Qwen2.5-7B-Instruct/samples_gsm8k_cot_zeroshot_2025-12-01T01-41-34.372070.jsonl",
        "big": QWEN_14B_MODEL_SAMPLES_PATH,
        # "big": GPT_5_MODEL_SAMPLES_PATH,
        "small": "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_qwen_familiy/gsm8k_cot_zeroshot/Qwen2.5-1.5B-Instruct_L8_BASELINE/Qwen__Qwen2.5-1.5B-Instruct/samples_gsm8k_cot_zeroshot_2025-12-11T02-46-20.289597.jsonl",
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        # "big": "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_qwen_familiy/gsm8k_cot_zeroshot/Qwen2.5-7B-Instruct_L8_BASELINE/Qwen__Qwen2.5-7B-Instruct/samples_gsm8k_cot_zeroshot_2025-12-01T01-41-34.372070.jsonl",
        "big": QWEN_14B_MODEL_SAMPLES_PATH,
        # "big": GPT_5_MODEL_SAMPLES_PATH,
        "small": "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_qwen_familiy/gsm8k_cot_zeroshot/Qwen2.5-3B-Instruct_L8_BASELINE/Qwen__Qwen2.5-3B-Instruct/samples_gsm8k_cot_zeroshot_2025-12-11T03-35-22.111548.jsonl",
    },
}



def get_exact_match(ex: dict):
        """
        Robustly extract exact_match from lm-eval sample json.
        Returns float (0.0/1.0) or None if missing.
        """
        if "exact_match" in ex:
            try:
                return float(ex["exact_match"])
            except Exception:
                pass
        # common nests
        for k in ["metrics", "metric", "results", "filtered", "eval", "scores"]:
            if k in ex and isinstance(ex[k], dict) and "exact_match" in ex[k]:
                try:
                    return float(ex[k]["exact_match"])
                except Exception:
                    pass
        # sometimes stored in "acc"/"exact_match,none" etc. (fallback: scan keys)
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
        output_hidden_states=True,
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

    # ===== 从后 200 个样本里统计 + 选正确的 =====
    TAIL_N = 200
    tail_ids = doc_ids[-TAIL_N:] if len(doc_ids) > TAIL_N else doc_ids

    # (A) 统计“后200个”里 exact_match 分布（按 big/teacher）
    cnt_1, cnt_0, cnt_missing = 0, 0, 0
    for did in tail_ids:
        em = get_exact_match(big_by_id[did])
        if em is None:
            cnt_missing += 1
        elif em >= 0.5:
            cnt_1 += 1
        else:
            cnt_0 += 1

    print(f"[Stats] last {len(tail_ids)} doc_ids: exact_match=1.0 -> {cnt_1}, 0.0 -> {cnt_0}, missing -> {cnt_missing}")

    stats_path = model_out_dir / "stats_last200.json"
    with open(stats_path, "w") as f:
        json.dump(
            {
                "n_checked": len(tail_ids),
                "exact_match_1": cnt_1,
                "exact_match_0": cnt_0,
                "exact_match_missing": cnt_missing,
                "note": "Counts computed from BIG/teacher samples (tail slice).",
            },
            f,
            indent=2,
        )
    # # ===== 选样本做 steering =====
    # SELECTED_IDS_PATH = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/vectors_Qwen_big_minus_small_selected_sample/Qwen_Qwen2.5-7B-Instruct/selected_doc_ids.json"
    # with open(SELECTED_IDS_PATH, "r") as f:
    #     selected_ids = json.load(f)
    # print("steering doc_ids (from selected sample file):", len(selected_ids))

    

    # (B) 只从“后200个”里挑 exact_match==1.0 的样本做 steering
    selected_ids = []
    for did in tail_ids:
        em = get_exact_match(big_by_id[did])
        if em is not None and em >= 0.5:
            selected_ids.append(did)

    # 你想要“前 MAX_EXAMPLES 个”就截断；也可以随机抽样（见下方可选）
    if MAX_EXAMPLES > 0:
        selected_ids = selected_ids[:MAX_EXAMPLES]

    print("steering doc_ids (tail exact_match==1.0):", len(selected_ids))

    docid_path = model_out_dir / "selected_doc_ids.json"
    with open(docid_path, "w") as f:
        json.dump(selected_ids, f, indent=2)

    # =====================
    # ===== PER LAYER =====
    # =====================

    for layer_idx in layer_list:
        print(f"  → Extracting layer {layer_idx}")
        vectors = []

        for did in tqdm(selected_ids, desc=f"Layer {layer_idx}"):

            ex_big   = big_by_id[did]
            ex_small = small_by_id[did]
            q = ex_big["doc"]["question"]

            prompt = qwen_chat_prompt(q)

            resp_big = ex_big["resps"][0][0]
            resp_small = ex_small["resps"][0][0]

            text1 = prompt + resp_big
            inputs1 = tokenizer(text1, return_tensors="pt", truncation=True).to(model.device)

            with torch.no_grad():
                out1 = model(**inputs1, use_cache=False)

            last_tok1 = inputs1["input_ids"].shape[1] - 1
            h1 = out1.hidden_states[layer_idx][0, last_tok1]
            
            text2 = prompt + resp_small
            inputs2 = tokenizer(text2, return_tensors="pt", truncation=True).to(model.device)

            with torch.no_grad():
                out2 = model(**inputs2, use_cache=False)

            last_tok2 = inputs2["input_ids"].shape[1] - 1
            h2 = out2.hidden_states[layer_idx][0, last_tok2]

            vectors.append((h1- h2).float().cpu()) # big - small

        V = torch.stack(vectors)  # (N, hidden)

        out_path = model_out_dir / f"layer_{layer_idx}.pt"
        torch.save(V, out_path)

        print(f"  ✔ Saved vectors: {out_path}, shape={V.shape}")

    del model
    torch.cuda.empty_cache()

print("\nAll models and layers processed.")
