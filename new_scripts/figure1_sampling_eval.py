#!/usr/bin/env python3
"""
Generate multiple sampled responses per GSM8K question for each Qwen model.
This enables within-question density vs correctness analysis for Figure 1.

Usage:
    CUDA_VISIBLE_DEVICES=0 python figure1_sampling_eval.py \
        --model Qwen/Qwen2.5-3B-Instruct --num_samples 8

Output: figure1_sampling_data/<model_sanitized>/gsm8k_samples.jsonl
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = (
    "Solve the following math problem. Present the final answer in the "
    "format: Final Answer: \\boxed{{your_answer}}.\n"
    "Prolbem: {question}\nAnswer:"
)

STOP_TOKENS = ["Q:", "</s>", "<|im_end|>", "<|eot_id|>",
               "<|start_header_id|>user<|end_header_id|>"]

OUT_ROOT = Path("/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/new_exps/figure1_sampling_data")


def extract_boxed_answer(text: str) -> str | None:
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if matches:
        return matches[-1].strip()
    match = re.search(r"Final Answer:\s*(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_gsm8k_gold(answer_str: str) -> str:
    match = re.search(r"####\s*(.+)", answer_str)
    if match:
        return match.group(1).strip().replace(",", "")
    return answer_str.strip()


def normalize_number(s: str) -> str:
    s = s.strip().rstrip(".").replace(",", "").replace("$", "").replace("%", "")
    s = s.replace(" ", "")
    try:
        return str(float(s))
    except ValueError:
        return s.lower()


def check_correct(pred_str: str | None, gold_str: str) -> bool:
    if pred_str is None:
        return False
    return normalize_number(pred_str) == normalize_number(gold_str)


def count_steps_and_density(text: str, tokenizer):
    steps = [s.strip() for s in text.split("\n\n") if s.strip()]
    if not steps:
        return 1, 0, 0, 0

    n_steps = len(steps)
    step_token_lens = []
    for step in steps:
        toks = tokenizer.encode(step, add_special_tokens=False)
        step_token_lens.append(len(toks))

    total_tokens = sum(step_token_lens)
    avg_tokens_per_step = float(np.mean(step_token_lens))
    density = total_tokens / n_steps  # ρ = N_tokens / N_steps

    return n_steps, total_tokens, avg_tokens_per_step, density


def build_chat_prompt(question: str, tokenizer) -> str:
    user_msg = PROMPT_TEMPLATE.format(question=question)
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": user_msg},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, num_samples, temperature, top_p,
                   max_new_tokens, device):
    expanded_prompts = []
    for p in prompts:
        expanded_prompts.extend([p] * num_samples)

    inputs = tokenizer(
        expanded_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(device)

    stop_ids = []
    for tok_str in STOP_TOKENS:
        ids = tokenizer.encode(tok_str, add_special_tokens=False)
        if len(ids) == 1:
            stop_ids.append(ids[0])
    if tokenizer.eos_token_id is not None:
        stop_ids.append(tokenizer.eos_token_id)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=stop_ids if stop_ids else None,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    input_len = inputs["input_ids"].shape[1]
    generated = outputs[:, input_len:]
    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)

    results = []
    for i, prompt in enumerate(prompts):
        sample_texts = texts[i * num_samples : (i + 1) * num_samples]
        results.append(sample_texts)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of GSM8K questions per batch (each expanded by num_samples)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of questions (for testing)")
    args = parser.parse_args()

    model_sanitized = args.model.replace("/", "_")
    out_dir = OUT_ROOT / model_sanitized
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gsm8k_samples.jsonl"

    print(f"Model: {args.model}")
    print(f"Samples per question: {args.num_samples}")
    print(f"Temperature: {args.temperature}, Top-p: {args.top_p}")
    print(f"Output: {out_path}")

    ds = load_dataset("gsm8k", "main", split="test")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"GSM8K test set: {len(ds)} questions")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    questions = [ex["question"] for ex in ds]
    gold_answers = [extract_gsm8k_gold(ex["answer"]) for ex in ds]

    prompts = [build_chat_prompt(q, tokenizer) for q in questions]

    total = len(prompts)
    bs = args.batch_size
    t0 = time.time()

    with open(out_path, "w") as fout:
        for batch_start in range(0, total, bs):
            batch_end = min(batch_start + bs, total)
            batch_prompts = prompts[batch_start:batch_end]
            batch_questions = questions[batch_start:batch_end]
            batch_golds = gold_answers[batch_start:batch_end]

            try:
                batch_results = generate_batch(
                    model, tokenizer, batch_prompts, args.num_samples,
                    args.temperature, args.top_p, args.max_new_tokens, device,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  OOM at batch {batch_start}, trying one at a time...")
                batch_results = []
                for p in batch_prompts:
                    r = generate_batch(
                        model, tokenizer, [p], args.num_samples,
                        args.temperature, args.top_p, args.max_new_tokens, device,
                    )
                    batch_results.append(r[0])

            for i, (question, gold, samples) in enumerate(
                zip(batch_questions, batch_golds, batch_results)
            ):
                doc_id = batch_start + i
                sample_records = []
                for s_idx, resp_text in enumerate(samples):
                    pred = extract_boxed_answer(resp_text)
                    correct = check_correct(pred, gold)
                    n_steps, total_toks, avg_tps, density = count_steps_and_density(
                        resp_text, tokenizer
                    )
                    sample_records.append({
                        "sample_idx": s_idx,
                        "response": resp_text,
                        "predicted_answer": pred,
                        "correct": correct,
                        "n_steps": n_steps,
                        "total_tokens": total_toks,
                        "avg_tokens_per_step": round(avg_tps, 2),
                        "density_rho": round(density, 2),
                    })

                record = {
                    "doc_id": doc_id,
                    "question": question,
                    "gold_answer": gold,
                    "model": args.model,
                    "num_samples": args.num_samples,
                    "temperature": args.temperature,
                    "samples": sample_records,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            elapsed = time.time() - t0
            done = batch_end
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            n_correct = sum(
                1 for rec in batch_results
                for resp in rec
                if check_correct(extract_boxed_answer(resp),
                                 batch_golds[batch_results.index(rec)])
            )
            print(
                f"  [{done}/{total}] {elapsed:.0f}s elapsed, "
                f"ETA {eta:.0f}s, {rate:.1f} q/s"
            )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Output: {out_path}")

    total_samples = 0
    total_correct = 0
    with open(out_path) as f:
        for line in f:
            obj = json.loads(line)
            for s in obj["samples"]:
                total_samples += 1
                if s["correct"]:
                    total_correct += 1
    print(f"Total samples: {total_samples}, Correct: {total_correct}, "
          f"pass@1 approx: {total_correct / total_samples:.3f}")


if __name__ == "__main__":
    main()
