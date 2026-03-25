#!/usr/bin/env python3
"""
Extract steering vectors from open-source model rewrites and evaluate on GSM8K.
Addresses reviewer concern: does GPT-5.1 inject hidden knowledge?
"""

import argparse
import json
import os
import subprocess
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from steering_vectors import train_steering_vector
from utils import qwen_chat_prompt

BASE = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement"


def load_rewritten_data(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_vector(model_name, data, layer_list, output_dir, num_examples=50):
    """Extract steering vector from rewritten pairs."""
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    ).eval()

    samples_used = data[:num_examples]
    print(f"Using {len(samples_used)} samples for vector extraction")

    training_samples = []
    for ex in samples_used:
        question = ex.get("question", ex.get("doc", {}).get("question", ""))
        prompt = qwen_chat_prompt(question)
        positive = prompt + ex["resp_after"]
        negative = prompt + ex["resp_before"]
        training_samples.append((positive, negative))

    print(f"Training steering vector across {len(layer_list)} layers...")
    sv = train_steering_vector(
        model=model,
        tokenizer=tokenizer,
        training_samples=training_samples,
        layers=layer_list,
        layer_type="decoder_block",
        move_to_cpu=True,
        read_token_index=-1,
        show_progress=True,
        batch_size=1,
    )

    save_path = os.path.join(output_dir, "steering_vector.pt")
    torch.save(sv, save_path)
    print(f"Saved steering vector to {save_path}")

    del model
    torch.cuda.empty_cache()
    return save_path


def evaluate_gsm8k(model_name, vec_path, layer, lam, output_dir, gpu=0):
    """Run lm_eval with steering vector."""
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "steer_hf",
        "--model_args", f"pretrained={model_name},dtype=float16,steer_layer={layer},steer_lambda={lam},steer_vec_path={vec_path},trust_remote_code=True",
        "--tasks", "gsm8k_cot_zeroshot_unified",
        "--batch_size", "8",
        "--num_fewshot", "0",
        "--output_path", output_dir,
        "--log_samples",
        "--trust_remote_code",
        "--gen_kwargs", "do_sample=False,temperature=0,max_gen_toks=2048",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"Running evaluation: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--rewriter", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--prompt_style", default="old")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--lam", type=float, default=4.0)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--num_examples", type=int, default=50)
    parser.add_argument("--skip_extract", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    model_safe = args.model.replace("/", "_")
    rewriter_safe = args.rewriter.replace("/", "_")

    rewrite_path = os.path.join(
        BASE, "opensource_rewrites", model_safe,
        f"rewritten_{args.prompt_style}_by_{rewriter_safe}.json"
    )

    vec_dir = os.path.join(
        BASE, "opensource_rewrites", model_safe,
        f"vectors_{args.num_examples}_{args.prompt_style}_by_{rewriter_safe}"
    )

    eval_dir = os.path.join(
        BASE, "opensource_rewrites", model_safe,
        f"eval_{args.prompt_style}_by_{rewriter_safe}_L{args.layer}_lam{args.lam}"
    )

    if not args.skip_extract:
        if not os.path.exists(rewrite_path):
            print(f"Error: rewrite data not found at {rewrite_path}")
            print("Run opensource_rewrite.py first.")
            sys.exit(1)

        data = load_rewritten_data(rewrite_path)
        layer_list = list(range(37))  # Qwen2.5-3B has 37 layers
        extract_vector(args.model, data, layer_list, vec_dir, args.num_examples)

    vec_path = os.path.join(vec_dir, "steering_vector.pt")
    if not args.skip_eval:
        evaluate_gsm8k(args.model, vec_path, args.layer, args.lam, eval_dir, args.gpu)


if __name__ == "__main__":
    main()
