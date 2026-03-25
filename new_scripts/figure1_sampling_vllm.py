#!/usr/bin/env python3
"""
vLLM-based sampling for Figure 1 analysis.
Uses tensor parallelism for multi-GPU inference on large models (14B+).

Usage:
    CUDA_VISIBLE_DEVICES=4,5 python figure1_sampling_vllm.py \
        --model Qwen/Qwen2.5-14B-Instruct --tensor_parallel_size 2
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

OUT_ROOT = Path("/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/new_exps/figure1_sampling_data")

PROMPT_TEMPLATE = (
    "Solve the following math problem. Present the final answer in the "
    "format: Final Answer: \\boxed{{your_answer}}.\n"
    "Prolbem: {question}\nAnswer:"
)


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
    step_token_lens = [
        len(tokenizer.encode(step, add_special_tokens=False))
        for step in steps
    ]
    total_tokens = sum(step_token_lens)
    avg_tokens_per_step = float(np.mean(step_token_lens))
    density = total_tokens / n_steps

    return n_steps, total_tokens, avg_tokens_per_step, density


def build_chat_prompt(question: str, tokenizer) -> str:
    user_msg = PROMPT_TEMPLATE.format(question=question)
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    model_sanitized = args.model.replace("/", "_")
    out_dir = OUT_ROOT / model_sanitized
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gsm8k_samples.jsonl"

    print(f"Model: {args.model}")
    print(f"Tensor parallel: {args.tensor_parallel_size} GPUs")
    print(f"Samples per question: {args.num_samples}")
    print(f"Temperature: {args.temperature}")
    print(f"Output: {out_path}")

    ds = load_dataset("gsm8k", "main", split="test")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"GSM8K test set: {len(ds)} questions")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("Loading vLLM engine...")
    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="float16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|eot_id|>", "Q:"],
    )

    questions = [ex["question"] for ex in ds]
    gold_answers = [extract_gsm8k_gold(ex["answer"]) for ex in ds]
    prompts = [build_chat_prompt(q, tokenizer) for q in questions]

    print(f"Generating {len(prompts)} x {args.num_samples} = {len(prompts) * args.num_samples} responses...")
    t0 = time.time()

    outputs = llm.generate(prompts, sampling_params)

    elapsed = time.time() - t0
    print(f"Generation done in {elapsed:.0f}s ({len(prompts) * args.num_samples / elapsed:.1f} samples/s)")

    with open(out_path, "w", encoding="utf-8") as fout:
        for doc_id, (output, question, gold) in enumerate(
            zip(outputs, questions, gold_answers)
        ):
            sample_records = []
            for s_idx, completion in enumerate(output.outputs):
                resp_text = completion.text
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

    total_samples = 0
    total_correct = 0
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            for s in obj["samples"]:
                total_samples += 1
                if s["correct"]:
                    total_correct += 1

    print(f"\nTotal samples: {total_samples}, Correct: {total_correct}, "
          f"pass@1 approx: {total_correct / total_samples:.3f}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
