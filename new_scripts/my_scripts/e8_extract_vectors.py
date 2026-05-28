#!/usr/bin/env python3
"""
E8 Calibration Size: Extract steering vectors for different calibration set sizes.
For each size N ∈ {1, 5, 10, 25, 50}, trains a steering vector using N sample pairs.
Saves both the vector (.pt) and per-layer norms (.json).

Usage:
    python e8_extract_vectors.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --mode GPT_REWRITE \
        --sizes 1 5 10 25 50 \
        --gpu 0

    python e8_extract_vectors.py \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --mode LARGE_MODEL \
        --rewrite_model meta-llama/Llama-3.1-8B-Instruct \
        --sizes 1 5 10 25 50 \
        --gpu 0
"""
import argparse
import json
import os
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from steering_vectors import train_steering_vector

BASE = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement"

DOC_TO_TEXT_TEMPLATE = (
    "Solve the following math problem. Present the final answer in the format: "
    "Final Answer: \\boxed{{your_answer}}.\n"
    "Prolbem: {question}\n"
    "Answer:"
)


def make_prompt(tokenizer, question):
    """Build prompt using tokenizer.apply_chat_template — matches lm_eval eval."""
    user_content = DOC_TO_TEXT_TEMPLATE.format(question=question)
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def get_exact_match(ex):
    if "exact_match" in ex:
        try:
            return float(ex["exact_match"])
        except (TypeError, ValueError):
            pass
    for k in ("metrics", "results", "scores"):
        if k in ex and isinstance(ex[k], dict) and "exact_match" in ex[k]:
            try:
                return float(ex[k]["exact_match"])
            except (TypeError, ValueError):
                pass
    return None


def load_samples(path):
    path = str(path)
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        for k in ("samples", "instances", "data"):
            if k in obj and isinstance(obj[k], list):
                return obj[k]
    return obj


def extract_for_size(model, tokenizer, model_name, selected_samples, n, layer_list, output_dir):
    """Extract and save a steering vector using exactly n samples."""
    samples = selected_samples[-n:] if len(selected_samples) >= n else selected_samples

    training_pairs = []
    for ex in samples:
        question = ex["doc"]["question"]
        prompt = make_prompt(tokenizer, question)
        training_pairs.append((
            prompt + ex["resp_after"],
            prompt + ex["resp_before"],
        ))

    print(f"  Training with {len(training_pairs)} pairs, {len(layer_list)} layers")

    sv = train_steering_vector(
        model=model,
        tokenizer=tokenizer,
        training_samples=training_pairs,
        layers=layer_list,
        layer_type="decoder_block",
        move_to_cpu=True,
        read_token_index=-1,
        show_progress=True,
        batch_size=1,
    )

    os.makedirs(output_dir, exist_ok=True)
    vec_path = os.path.join(output_dir, "steering_vector.pt")
    torch.save(sv, vec_path)

    norms = {}
    for layer_idx, vec in sv.layer_activations.items():
        norms[int(layer_idx)] = round(vec.float().norm().item(), 6)

    norms_path = os.path.join(output_dir, "vector_norms.json")
    with open(norms_path, "w") as f:
        json.dump(norms, f, indent=2)

    print(f"  Saved vector -> {vec_path}")
    print(f"  Saved norms  -> {norms_path}")
    return vec_path, norms


def main():
    parser = argparse.ArgumentParser(
        description="E8 Calibration Size: extract steering vectors for varying N"
    )
    parser.add_argument("--model", required=True,
                        help="HF model name, e.g. Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--mode", required=True, choices=["GPT_REWRITE", "LARGE_MODEL"])
    parser.add_argument("--rewrite_model", default=None,
                        help="Large rewrite model (LARGE_MODEL mode only)")
    parser.add_argument("--prompt_style", default="old",
                        help="Prompt style (GPT_REWRITE mode only)")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1, 5, 10, 25, 50],
                        help="Calibration set sizes to test")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_base", default=None,
                        help="Override base output directory")
    args = parser.parse_args()

    model_safe = args.model.replace("/", "_")

    if args.mode == "GPT_REWRITE":
        sample_path = os.path.join(
            BASE, "exps", "gpt_rewrites_unified_new", model_safe,
            f"rewritten_{args.prompt_style}.json"
        )
    else:
        if not args.rewrite_model:
            parser.error("--rewrite_model is required for LARGE_MODEL mode")
        rewrite_safe = args.rewrite_model.replace("/", "_")
        sample_path = os.path.join(
            BASE, "exps", "large_model_rewrites_unified_new", model_safe,
            f"{rewrite_safe}_paired_responses.json"
        )

    if not os.path.exists(sample_path):
        print(f"[ERROR] Sample file not found: {sample_path}")
        sys.exit(1)

    print(f"Mode:    {args.mode}")
    print(f"Model:   {args.model}")
    print(f"Samples: {sample_path}")
    print(f"Sizes:   {args.sizes}")

    samples = load_samples(sample_path)
    by_id = {ex["doc_id"]: ex for ex in samples if "doc_id" in ex}
    doc_ids = sorted(by_id.keys())

    max_n = max(args.sizes)
    selected = []
    for did in reversed(doc_ids):
        if get_exact_match(by_id[did]) == 1.0:
            selected.append(by_id[did])
            if len(selected) >= max_n:
                break
    selected = list(reversed(selected))
    print(f"Available EM=1 samples: {len(selected)} (need up to {max_n})")

    if not selected:
        print("[ERROR] No EM=1 samples found!")
        sys.exit(1)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    num_layers = config.num_hidden_layers
    layer_list = list(range(num_layers))
    print(f"Model has {num_layers} layers, extracting all.")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    ).eval()

    if args.output_base:
        out_base = args.output_base
    else:
        if args.mode == "GPT_REWRITE":
            out_base = os.path.join(
                BASE, "calibration_ablation", model_safe, "GPT_REWRITE"
            )
        else:
            rewrite_safe = args.rewrite_model.replace("/", "_")
            out_base = os.path.join(
                BASE, "calibration_ablation", model_safe,
                f"LARGE_MODEL_{rewrite_safe}",
            )

    all_norms = {}
    for n in args.sizes:
        print(f"\n{'=' * 60}")
        print(f" N = {n}")
        print(f"{'=' * 60}")
        vec_dir = os.path.join(out_base, f"vectors_N{n}")
        _, norms = extract_for_size(
            model, tokenizer, args.model, selected, n, layer_list, vec_dir
        )
        all_norms[n] = norms

    del model
    torch.cuda.empty_cache()

    summary_path = os.path.join(out_base, "extraction_summary.json")
    os.makedirs(out_base, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "model": args.model,
            "mode": args.mode,
            "sizes": args.sizes,
            "norms": {str(k): v for k, v in all_norms.items()},
        }, f, indent=2)
    print(f"\nAll extractions complete. Summary: {summary_path}")


if __name__ == "__main__":
    main()
