# run_pipeline.py
import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from manual_eval_gsm8k import run_manual_eval_and_select


# =====================
# ===== DEFAULT CONFIG
# =====================

MODEL_NAME_TO_LAYER_INDEX = {
    "Qwen/Qwen2.5-7B-Instruct": [14, 24, 28],
    "Qwen/Qwen2.5-1.5B-Instruct": [14, 24, 28],
    "Qwen/Qwen2.5-3B-Instruct": [18, 32, 36],
}

MODEL_NAME_TO_SAMPLE_PATHS = {
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


def load_jsonl_map(path: str) -> Dict[int, dict]:
    by_id = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            by_id[int(ex["doc_id"])] = ex
    return by_id


def extract_vectors_for_model(
    model_name: str,
    layer_list: List[int],
    big_path: str,
    small_path: str,
    out_dir: Path,
    selected_ids: List[int],
    dtype: str = "bfloat16",
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # load data
    big_by_id = load_jsonl_map(big_path)
    small_by_id = load_jsonl_map(small_path)

    common = sorted(set(big_by_id) & set(small_by_id))
    selected_ids = [i for i in selected_ids if i in common]
    if len(selected_ids) == 0:
        raise ValueError(f"No selected_ids exist in the intersection for model={model_name}")

    # save used ids (after intersection)
    with open(out_dir / "selected_doc_ids_used.json", "w") as f:
        json.dump(selected_ids, f, indent=2)

    # model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        output_hidden_states=True,
    ).eval()

    # extract
    for layer_idx in layer_list:
        print(f"  → Extracting layer {layer_idx}")
        vectors = []

        for did in tqdm(selected_ids, desc=f"{model_name} L{layer_idx}"):
            ex_big = big_by_id[did]
            ex_small = small_by_id[did]

            # 注意：这里沿用你原逻辑：prompt + resp
            text_big = ex_big["arguments"]["gen_args_0"]["arg_0"] + ex_big["resps"][0][0]
            inp_big = tokenizer(text_big, return_tensors="pt", truncation=True).to(model.device)
            with torch.no_grad():
                out_big = model(**inp_big, use_cache=False)
            h_big = out_big.hidden_states[layer_idx][0, inp_big["input_ids"].shape[1] - 1]

            text_small = ex_small["arguments"]["gen_args_0"]["arg_0"] + ex_small["resps"][0][0]
            inp_small = tokenizer(text_small, return_tensors="pt", truncation=True).to(model.device)
            with torch.no_grad():
                out_small = model(**inp_small, use_cache=False)
            h_small = out_small.hidden_states[layer_idx][0, inp_small["input_ids"].shape[1] - 1]

            # ✅ big - small：配合你 steer_hf 里 y += lambda*w, 用正 lambda 推向 big
            vectors.append((h_big - h_small).float().cpu())

        V = torch.stack(vectors)  # (N, hidden)
        out_path = out_dir / f"layer_{layer_idx}.pt"
        torch.save(V, out_path)
        print(f"  ✔ Saved vectors: {out_path}, shape={tuple(V.shape)}")

    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eval", "extract", "both"], default="both",
                    help="eval: manual exact_match selection; extract: vectors only; both: eval then extract")
    ap.add_argument("--models", nargs="*", default=list(MODEL_NAME_TO_LAYER_INDEX.keys()),
                    help="subset of model names to run")
    ap.add_argument("--out-root", type=str, default="./vectors_14b_big_minus_small_selected_sample",
                    help="root output dir")
    ap.add_argument("--select-source", choices=["big", "small"], default="big",
                    help="use which model outputs to compute manual EM for selection")
    ap.add_argument("--max-examples", type=int, default=100)
    ap.add_argument("--first-n", type=int, default=100)
    ap.add_argument("--overwrite-selection", action="store_true")
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        if model_name not in MODEL_NAME_TO_LAYER_INDEX:
            raise ValueError(f"Unknown model_name: {model_name}")

        layer_list = MODEL_NAME_TO_LAYER_INDEX[model_name]
        paths = MODEL_NAME_TO_SAMPLE_PATHS[model_name]
        big_path, small_path = paths["big"], paths["small"]

        model_tag = model_name.replace("/", "_")
        model_out_dir = out_root / model_tag
        model_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n========== {model_name} ==========")

        selected_ids = None

        # ---- manual eval selection ----
        if args.mode in ("eval", "both"):
            selected_ids, stats = run_manual_eval_and_select(
                big_path=big_path,
                small_path=small_path,
                out_dir=model_out_dir,
                select_source=args.select_source,
                max_examples=args.max_examples,
                first_n=args.first_n,
                overwrite=args.overwrite_selection,
            )
            print(f"[Eval] first{args.first_n} stats:", stats)
            print(f"[Eval] selected correct ids: {len(selected_ids)}")

        # ---- extract vectors ----
        if args.mode in ("extract", "both"):
            # 如果只 extract，但没 eval：就读已有 selection
            if selected_ids is None:
                sel_path = model_out_dir / "selected_doc_ids.json"
                if not sel_path.exists():
                    raise FileNotFoundError(
                        f"{sel_path} not found. Run with --mode eval or --mode both first, "
                        f"or provide selected_doc_ids.json."
                    )
                with open(sel_path, "r") as f:
                    selected_ids = json.load(f)

            extract_vectors_for_model(
                model_name=model_name,
                layer_list=layer_list,
                big_path=big_path,
                small_path=small_path,
                out_dir=model_out_dir,
                selected_ids=selected_ids,
                dtype=args.dtype,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
