import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path

# =====================
# ===== CONFIG ========
# =====================

model_name_to_layer_index = {
    "Qwen/Qwen2.5-7B-Instruct": [14, 24, 28],
    "Qwen/Qwen2.5-1.5B-Instruct": [14, 24, 28],
    "Qwen/Qwen2.5-3B-Instruct": [18, 32, 36],
}


model_name_to_sample_paths = {
    "Qwen/Qwen2.5-7B-Instruct": {
        "big": "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_qwen_familiy/gsm8k_cot_zeroshot/Qwen2.5-14B-Instruct_L8_BASELINE/Qwen__Qwen2.5-14B-Instruct/samples_gsm8k_cot_zeroshot_2025-12-11T09-28-44.120949.jsonl",
        "small": "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_qwen_familiy/gsm8k_cot_zeroshot/Qwen2.5-7B-Instruct_L8_BASELINE/Qwen__Qwen2.5-7B-Instruct/samples_gsm8k_cot_zeroshot_2025-12-01T01-41-34.372070.jsonl",
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "big": "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_qwen_familiy/gsm8k_cot_zeroshot/Qwen2.5-14B-Instruct_L8_BASELINE/Qwen__Qwen2.5-14B-Instruct/samples_gsm8k_cot_zeroshot_2025-12-11T09-28-44.120949.jsonl",
        "small": "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_qwen_familiy/gsm8k_cot_zeroshot/Qwen2.5-1.5B-Instruct_L8_BASELINE/Qwen__Qwen2.5-1.5B-Instruct/samples_gsm8k_cot_zeroshot_2025-12-11T02-46-20.289597.jsonl",
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "big": "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_qwen_familiy/gsm8k_cot_zeroshot/Qwen2.5-14B-Instruct_L8_BASELINE/Qwen__Qwen2.5-14B-Instruct/samples_gsm8k_cot_zeroshot_2025-12-11T09-28-44.120949.jsonl",
        "small": "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_qwen_familiy/gsm8k_cot_zeroshot/Qwen2.5-3B-Instruct_L8_BASELINE/Qwen__Qwen2.5-3B-Instruct/samples_gsm8k_cot_zeroshot_2025-12-11T03-35-22.111548.jsonl",
    },
}

STEER_RATIO = 0.1
MAX_EXAMPLES = 100

root_out_dir = Path("./vectors")
root_out_dir.mkdir(exist_ok=True)

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
    k = int(len(doc_ids) * STEER_RATIO)
    selected_ids = doc_ids[:k]

    if MAX_EXAMPLES > 0:
        selected_ids = selected_ids[:MAX_EXAMPLES]

    print("steering doc_ids:", len(selected_ids))

    # ---- save doc ids once per model ----
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

            text1 = ex_big["arguments"]["gen_args_0"]["arg_0"] + ex_big["resps"][0][0]
            inputs1 = tokenizer(text1, return_tensors="pt", truncation=True).to(model.device)

            with torch.no_grad():
                out1 = model(**inputs1, use_cache=False)

            last_tok1 = inputs1["input_ids"].shape[1] - 1
            h1 = out1.hidden_states[layer_idx][0, last_tok1]

            text2 = ex_small["arguments"]["gen_args_0"]["arg_0"] + ex_small["resps"][0][0]
            inputs2 = tokenizer(text2, return_tensors="pt", truncation=True).to(model.device)

            with torch.no_grad():
                out2 = model(**inputs2, use_cache=False)

            last_tok2 = inputs2["input_ids"].shape[1] - 1
            h2 = out2.hidden_states[layer_idx][0, last_tok2]

            vectors.append((h2 - h1).float().cpu())

        V = torch.stack(vectors)  # (N, hidden)

        out_path = model_out_dir / f"layer_{layer_idx}.pt"
        torch.save(V, out_path)

        print(f"  ✔ Saved vectors: {out_path}, shape={V.shape}")

    del model
    torch.cuda.empty_cache()

print("\nAll models and layers processed.")
