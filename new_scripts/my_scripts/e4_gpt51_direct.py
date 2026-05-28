#!/usr/bin/env python3
"""
E4: GPT-5.1 Direct Answer Control Experiment.

Instead of rewriting the Qwen-3B response, GPT-5.1 solves the question
from scratch. The GPT-5.1 answer becomes `resp_after`, the original
Qwen-3B response stays as `resp_before`.

Usage:
    python new_scripts/my_scripts/e4_gpt51_direct.py
"""

import json
import os
import sys
import time
from copy import deepcopy
from tqdm import tqdm

BASE = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement"
SOURCE_PATH = os.path.join(
    BASE, "exps", "gpt_rewrites_unified_new",
    "Qwen_Qwen2.5-3B-Instruct", "rewritten_old.json",
)
OUTPUT_DIR = os.path.join(BASE, "control_experiments", "Qwen_Qwen2.5-3B-Instruct")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "rewritten_e4_gpt51_direct.json")

SOLVE_PROMPT = (
    "Solve the following math problem. Present the final answer in the format:\n"
    "Final Answer: \\boxed{{your_answer}}.\n"
    "Problem: {question}\n"
    "Answer:"
)


def call_llm(client, model, prompt, max_tokens=8192, temperature=0.0, retries=4):
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            wait = 1.5 * (2 ** attempt)
            print(f"  [retry {attempt+1}/{retries}] {e} — waiting {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"API failed after {retries} retries: {last_err}")


def main():
    # Load env
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE, ".env"))

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("BASE_URL")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY in .env", file=sys.stderr)
        sys.exit(2)

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    model = "gpt-5.1"

    # Load source data — same selection as E0-E3
    data = json.load(open(SOURCE_PATH))
    em1 = [ex for ex in data if ex.get("exact_match", 0) == 1.0]
    selected = em1[-50:]
    print(f"Loaded {len(data)} samples, {len(em1)} EM=1, selected {len(selected)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Resume support: load existing results if any
    existing = []
    if os.path.exists(OUTPUT_FILE):
        existing = json.load(open(OUTPUT_FILE))
        print(f"Resuming from {len(existing)} existing results")

    done_ids = {ex["doc_id"] for ex in existing}
    results = list(existing)

    for ex in tqdm(selected, desc="E4 GPT-5.1 Direct"):
        if ex["doc_id"] in done_ids:
            continue

        question = ex["doc"]["question"]
        prompt = SOLVE_PROMPT.format(question=question)
        gpt_answer = call_llm(client, model, prompt)

        new_ex = deepcopy(ex)
        new_ex["resp_after"] = gpt_answer
        new_ex["resp_before"] = ex["resp_before"]
        new_ex["resp_rewrite_style"] = "e4_gpt51_direct"
        new_ex["resp_rewrite_ok"] = True
        results.append(new_ex)

        # Save incrementally every 5 samples
        if len(results) % 5 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} samples to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
