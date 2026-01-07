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
    # "Qwen/Qwen2.5-7B-Instruct": [14, 24, 28],
    "Qwen/Qwen2.5-1.5B-Instruct": [14, 24, 28],
    # "Qwen/Qwen2.5-3B-Instruct": [18, 32, 36],
}
NUM_EXAMPLES = 50

root_out_dir = Path(f"./vectors_less_steps_{NUM_EXAMPLES}")
root_out_dir.mkdir(exist_ok=True)
QWEN_1_5B_MODEL_LESS_STEPS_SAMPLES_PATH = "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_qwen_familiy/gsm8k_cot_zeroshot/Qwen2.5-1.5B-Instruct_L8_BASELINE/Qwen__Qwen2.5-1.5B-Instruct/samples_gsm8k_cot_zeroshot_rewritten.json"
model_name_to_sample_paths = {
    "Qwen/Qwen2.5-1.5B-Instruct": QWEN_1_5B_MODEL_LESS_STEPS_SAMPLES_PATH
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


def load_samples(path: str):
    """Load lm-eval samples from .json (list) or .jsonl (lines)."""
    path = str(path)
    if path.endswith(".jsonl"):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    else:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        # some lm-eval dumps wrap as {"samples":[...]} or {"instances":[...]}
        if isinstance(obj, dict):
            for k in ["samples", "instances", "data"]:
                if k in obj and isinstance(obj[k], list):
                    return obj[k]
        if isinstance(obj, list):
            return obj
        raise ValueError(f"Unrecognized json format in {path}")


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
    sample_path = model_name_to_sample_paths[model_name]
    samples = load_samples(sample_path)
        
    # build by_id
    by_id = {}
    for ex in samples:
        did = ex.get("doc_id")
        if did is None:
            continue
        by_id[did] = ex

    doc_ids = sorted(by_id.keys())
    # 从后往前挑 NUM_EXAMPLES 个 exact_match == 1.0 的样本
    selected_ids = []
    for did in reversed(doc_ids):  # 从最后一个 doc_id 开始往前扫
        em = get_exact_match(by_id[did])
        if em is not None and float(em) == 1.0:
            selected_ids.append(did)
            if NUM_EXAMPLES > 0 and len(selected_ids) >= NUM_EXAMPLES:
                break

    # selected_ids 当前是“从后往前收集”的顺序；如果你希望保持原 doc_id 升序/一致性，就反转一下
    selected_ids = list(reversed(selected_ids))

    print(f"steering doc_ids (from tail, exact_match==1.0): {len(selected_ids)} / requested {NUM_EXAMPLES}")

    docid_path = model_out_dir / "selected_doc_ids.json"
    with open(docid_path, "w", encoding="utf-8") as f:
        json.dump(selected_ids, f, indent=2)

    # =====================
    # ===== PER LAYER =====
    # =====================

    for layer_idx in layer_list:
        print(f"  → Extracting layer {layer_idx}")
        vectors = []

        for did in tqdm(selected_ids, desc=f"Layer {layer_idx}"):

            ex = by_id[did]
            q = ex["doc"]["question"]

            prompt = qwen_chat_prompt(q)

            resp_original, resp_rewritten = ex["resp_before"], ex["resp_after"]

            text_rewritten = prompt + resp_rewritten
            inputs_r = tokenizer(text_rewritten, return_tensors="pt", truncation=True).to(model.device)
            with torch.no_grad():
                out_r = model(**inputs_r, use_cache=False)
            last_tok_r = inputs_r["input_ids"].shape[1] - 1
            h_r = out_r.hidden_states[layer_idx][0, last_tok_r]
            
            # ---- original ----
            text_original = prompt + resp_original
            inputs_o = tokenizer(text_original, return_tensors="pt", truncation=True).to(model.device)
            with torch.no_grad():
                out_o = model(**inputs_o, use_cache=False)
            last_tok_o = inputs_o["input_ids"].shape[1] - 1
            h_o = out_o.hidden_states[layer_idx][0, last_tok_o]

            # vector = rewritten - original
            vectors.append((h_r - h_o).float().cpu())

        V = torch.stack(vectors)  # (N, hidden)

        out_path = model_out_dir / f"layer_{layer_idx}.pt"
        torch.save(V, out_path)

        print(f"  ✔ Saved vectors: {out_path}, shape={V.shape}")

    del model
    torch.cuda.empty_cache()

print("\nAll models and layers processed.")
