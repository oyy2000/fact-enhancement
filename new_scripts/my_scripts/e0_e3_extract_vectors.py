#!/usr/bin/env python3
"""
E0-E3: Extract steering vectors for each ablation experiment.

For each experiment variant, loads the paired data, filters EM=1 samples,
constructs (prompt + resp_after, prompt + resp_before) pairs, and trains
a steering vector across all layers of Qwen2.5-3B-Instruct (37 layers).

Usage:
    python e0_e3_extract_vectors.py --experiment e0 --gpu 0
    python e0_e3_extract_vectors.py --experiment e1 --gpu 1
    python e0_e3_extract_vectors.py --experiment e2 --gpu 2
    python e0_e3_extract_vectors.py --experiment e3 --gpu 3
    python e0_e3_extract_vectors.py --experiment e3.2 --gpu 4

    # Or extract all at once (sequentially on one GPU):
    python e0_e3_extract_vectors.py --experiment all --gpu 0
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
CONTROL_DIR = os.path.join(BASE, "control_experiments", "Qwen_Qwen2.5-3B-Instruct")
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

DOC_TO_TEXT_TEMPLATE = (
    "Solve the following math problem. Present the final answer in the format: "
    "Final Answer: \\boxed{{your_answer}}.\n"
    "Prolbem: {question}\n"
    "Answer:"
)

EXPERIMENT_TO_FILE = {
    "e0":   "rewritten_e0_paraphrase.json",
    "e1":   "rewritten_e1_random_step_compress.json",
    "e2":   "rewritten_e2_dense_incorrect.json",
    "e3":   "rewritten_e3_rule_based.json",
    "e3.2": "rewritten_e3_2_gpt54mini.json",
    "e4":   "rewritten_e4_gpt51_direct.json",
}


def make_prompt(tokenizer, question):
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
    return None


def extract_for_experiment(model, tokenizer, experiment_name, data_path,
                           layer_list, num_examples=50):
    """Extract and save steering vector for one experiment."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    em1 = [ex for ex in data if get_exact_match(ex) == 1.0]
    selected = em1[-num_examples:]
    print(f"  {experiment_name}: {len(selected)} EM=1 samples (of {len(em1)} available)")

    if not selected:
        print(f"  [ERROR] No EM=1 samples for {experiment_name}!")
        return None

    training_pairs = []
    for ex in selected:
        question = ex["doc"]["question"]
        prompt = make_prompt(tokenizer, question)
        training_pairs.append((
            prompt + ex["resp_after"],
            prompt + ex["resp_before"],
        ))

    print(f"  Training with {len(training_pairs)} pairs across {len(layer_list)} layers...")

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

    out_dir = os.path.join(CONTROL_DIR, f"vectors_{experiment_name}")
    os.makedirs(out_dir, exist_ok=True)

    vec_path = os.path.join(out_dir, "steering_vector.pt")
    torch.save(sv, vec_path)
    print(f"  Saved vector -> {vec_path}")

    norms = {}
    for layer_idx, vec in sv.layer_activations.items():
        norms[int(layer_idx)] = round(vec.float().norm().item(), 6)

    norms_path = os.path.join(out_dir, "vector_norms.json")
    with open(norms_path, "w") as f:
        json.dump(norms, f, indent=2)
    print(f"  Saved norms  -> {norms_path}")

    if 6 in norms:
        print(f"  Layer 6 norm = {norms[6]:.4f}")

    return vec_path


def main():
    parser = argparse.ArgumentParser(description="E0-E3 Steering Vector Extraction")
    parser.add_argument("--experiment", required=True,
                        choices=list(EXPERIMENT_TO_FILE.keys()) + ["all"])
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_examples", type=int, default=50)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    experiments = (
        list(EXPERIMENT_TO_FILE.keys()) if args.experiment == "all"
        else [args.experiment]
    )

    for exp_name in experiments:
        data_path = os.path.join(CONTROL_DIR, EXPERIMENT_TO_FILE[exp_name])
        if not os.path.exists(data_path):
            print(f"[SKIP] Data file not found for {exp_name}: {data_path}")
            continue

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    num_layers = config.num_hidden_layers
    layer_list = list(range(num_layers))
    print(f"Model: {args.model} ({num_layers} layers)")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    ).eval()

    for exp_name in experiments:
        data_path = os.path.join(CONTROL_DIR, EXPERIMENT_TO_FILE[exp_name])
        if not os.path.exists(data_path):
            continue
        print(f"\n{'='*60}")
        print(f" Extracting: {exp_name}")
        print(f"{'='*60}")
        extract_for_experiment(
            model, tokenizer, exp_name, data_path, layer_list, args.num_examples
        )

    del model
    torch.cuda.empty_cache()
    print("\nAll extractions complete.")


if __name__ == "__main__":
    main()
