#!/usr/bin/env python3
"""
E0-E3 Ablation Data Preparation.

Generates modified (resp_before, resp_after) pairs for each ablation:
  E0   - Random Paraphrase (GPT-5.1, semantics-preserving, same N_steps)
  E1   - Random Step Compression (GPT-5.1, randomly pick 2 steps to shorten)
  E2   - Dense but Incorrect (corrupt 2 calc results in existing dense traces)
  E3   - Rule-Based Rewriting (heuristic merge, no LLM)
  E3.2 - GPT-5-mini dense rewriting (same prompt as DenseSteer "old")

Usage:
    python e0_e3_prepare_data.py --experiment e0
    python e0_e3_prepare_data.py --experiment e1
    python e0_e3_prepare_data.py --experiment e2   # no API key needed
    python e0_e3_prepare_data.py --experiment e3   # no API key needed
    python e0_e3_prepare_data.py --experiment e3.2
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

from tqdm import tqdm

BASE = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement"
SOURCE_PATH = os.path.join(
    BASE, "exps", "gpt_rewrites_unified_new",
    "Qwen_Qwen2.5-3B-Instruct", "rewritten_old.json",
)
OUTPUT_DIR = os.path.join(BASE, "control_experiments", "Qwen_Qwen2.5-3B-Instruct")

EXPERIMENTS_NEEDING_API = {"e0", "e1", "e3.2"}


# ─── Text splitting (from utils.py) ──────────────────────────

def split_double_newline(text):
    if not text:
        return []
    parts = text.split("\n\n")
    steps = []
    for i, p in enumerate(parts):
        if not p.strip():
            continue
        if i < len(parts) - 1:
            steps.append(p + "\n\n")
        else:
            steps.append(p)
    return steps


def split_single_newline_robust(text):
    if not text:
        return []
    parts = re.split(r"(\n+)", text)
    steps = []
    for i in range(0, len(parts), 2):
        step = parts[i]
        if i + 1 < len(parts):
            step += parts[i + 1]
        if step.strip():
            steps.append(step)
    return steps


def split_auto(text):
    if "\n\n" in text:
        return split_double_newline(text)
    return split_single_newline_robust(text)


# ─── OpenAI helper ───────────────────────────────────────────

def call_llm(client, model, prompt, max_tokens=8192, temperature=0.0, retries=4):
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (2 ** attempt))
    raise RuntimeError(f"API failed after {retries} retries: {last_err}")


# ─── E0: Random Paraphrase ──────────────────────────────────

def build_prompt_e0_paraphrase(question: str, original_resp: str) -> str:
    return f"""You will paraphrase the solution below. The goal is to change ONLY the surface wording (synonyms, word order, connectors) while keeping the meaning, structure, and reasoning density IDENTICAL.

Hard constraints:
- Keep the EXACT same number of reasoning steps / paragraphs. Do NOT merge, split, add, or remove any step.
- Do NOT change any numbers, equations, calculations, or final answers.
- Do NOT add new reasoning or remove existing reasoning.
- Change vocabulary: replace words with synonyms, rephrase sentences, adjust syntax, swap connectors (e.g. "Therefore" → "Hence", "First" → "To begin with").
- Preserve special markers like "<<a=b>>" exactly.
- Preserve the overall formatting (newlines, bullet points, step labels if any).
- Output plain text only. Output ONLY the paraphrased solution, no explanations.

Question:
{question}

Original solution:
{original_resp}

Paraphrased solution:
"""


def prepare_e0(data, client, model, num_examples=50):
    em1 = [ex for ex in data if ex.get("exact_match", 0) == 1.0]
    selected = em1[-num_examples:]
    print(f"E0: {len(selected)} EM=1 samples selected for paraphrase")

    results = []
    for ex in tqdm(selected, desc="E0 Paraphrase"):
        new_ex = deepcopy(ex)
        original = ex["resp_before"]
        question = ex["doc"]["question"]
        prompt = build_prompt_e0_paraphrase(question, original)
        paraphrased = call_llm(client, model, prompt)
        new_ex["resp_after"] = paraphrased
        new_ex["resp_before"] = original
        new_ex["resp_rewrite_style"] = "e0_paraphrase"
        new_ex["resp_rewrite_ok"] = True
        results.append(new_ex)
    return results


# ─── E1: Random Step Compression ────────────────────────────

def build_prompt_e1_compress_steps(question: str, original_resp: str,
                                    step_indices: list, steps: list) -> str:
    step_texts = "\n".join(
        f"[Step {idx}]: {steps[idx].strip()}" for idx in step_indices
    )
    return f"""You will rewrite ONLY the specified steps of the solution below to be shorter and more concise, while strictly preserving the semantics and correctness.

The full solution is provided for context, but you must ONLY rewrite the indicated steps.

Hard constraints:
- Rewrite ONLY the steps listed below. Return the FULL solution with the other steps UNCHANGED.
- Make the indicated steps shorter: remove filler words, combine clauses, use more direct phrasing.
- Do NOT change any numbers, equations, calculations, or final answers.
- Do NOT remove any logical step or skip any reasoning.
- Do NOT change the total number of steps/paragraphs.
- Preserve special markers like "<<a=b>>" exactly.
- Output plain text only. Output ONLY the rewritten full solution.

Question:
{question}

Steps to rewrite (make shorter):
{step_texts}

Full original solution:
{original_resp}

Rewritten solution (with indicated steps compressed):
"""


def prepare_e1(data, client, model, num_examples=50, seed=42):
    rng = random.Random(seed)
    em1 = [ex for ex in data if ex.get("exact_match", 0) == 1.0]
    selected = em1[-num_examples:]
    print(f"E1: {len(selected)} EM=1 samples selected for random step compression")

    results = []
    for ex in tqdm(selected, desc="E1 Random Step Compress"):
        new_ex = deepcopy(ex)
        original = ex["resp_before"]
        question = ex["doc"]["question"]
        steps = split_auto(original)

        n_steps = len(steps)
        if n_steps <= 2:
            indices = list(range(n_steps))
        else:
            indices = sorted(rng.sample(range(n_steps), min(2, n_steps)))

        prompt = build_prompt_e1_compress_steps(question, original, indices, steps)
        compressed = call_llm(client, model, prompt)
        new_ex["resp_after"] = compressed
        new_ex["resp_before"] = original
        new_ex["resp_rewrite_style"] = "e1_random_step_compress"
        new_ex["resp_rewrite_ok"] = True
        new_ex["compressed_step_indices"] = indices
        results.append(new_ex)
    return results


# ─── E2: Dense but Incorrect ────────────────────────────────

_CALC_PATTERNS = [
    re.compile(r"(<<[^=]+=\s*)(\d+(?:\.\d+)?)(>>)"),
    re.compile(r"(=\s*\$?)(\d+(?:,\d{3})*(?:\.\d+)?)(\b)"),
    re.compile(r"(=\s*)(\d+(?:,\d{3})*(?:\.\d+)?)(\s)"),
]


def _find_all_numbers(text):
    """Find (start, end, value_str, pattern_idx) for corruptible numbers."""
    hits = []
    for pidx, pat in enumerate(_CALC_PATTERNS):
        for m in pat.finditer(text):
            val_str = m.group(2).replace(",", "")
            try:
                float(val_str)
            except ValueError:
                continue
            hits.append((m.start(2), m.end(2), m.group(2), pidx))
    seen_positions = set()
    unique = []
    for h in hits:
        if h[0] not in seen_positions:
            seen_positions.add(h[0])
            unique.append(h)
    return unique


def corrupt_number(val_str, rng):
    """Corrupt a number string to a plausible but wrong value."""
    clean = val_str.replace(",", "")
    try:
        val = float(clean)
    except ValueError:
        return val_str

    if val == 0:
        new_val = rng.choice([1, 2, 3, 5])
    elif abs(val) < 1:
        new_val = val + rng.choice([-0.5, 0.3, 0.7, -0.2])
    else:
        perturbations = [
            val + rng.randint(1, max(3, int(abs(val) * 0.3))),
            val - rng.randint(1, max(3, int(abs(val) * 0.3))),
            val * rng.choice([1.1, 1.25, 0.8, 0.9, 1.5]),
        ]
        new_val = rng.choice(perturbations)
        if new_val == val:
            new_val = val + rng.choice([1, -1, 2, -2])

    if "." not in clean:
        new_val = int(round(new_val))
    else:
        new_val = round(new_val, len(clean.split(".")[-1]))

    result = str(new_val)
    if "," in val_str and abs(new_val) >= 1000:
        parts = []
        integer_part = str(int(abs(new_val)))
        for i, ch in enumerate(reversed(integer_part)):
            if i > 0 and i % 3 == 0:
                parts.append(",")
            parts.append(ch)
        result = "".join(reversed(parts))
        if new_val < 0:
            result = "-" + result
    return result


def prepare_e2(data, num_examples=50, seed=42):
    rng = random.Random(seed)
    em1 = [ex for ex in data if ex.get("exact_match", 0) == 1.0]
    selected = em1[-num_examples:]
    print(f"E2: {len(selected)} EM=1 samples selected for dense-incorrect corruption")

    results = []
    total_corrupted = 0
    for ex in tqdm(selected, desc="E2 Dense Incorrect"):
        new_ex = deepcopy(ex)
        dense_text = ex["resp_after"]  # start from correct dense traces
        hits = _find_all_numbers(dense_text)

        n_corrupt = min(2, len(hits))
        if n_corrupt > 0:
            chosen = sorted(rng.sample(hits, n_corrupt), key=lambda h: h[0], reverse=True)
            corrupted = dense_text
            corruption_log = []
            for start, end, val_str, pidx in chosen:
                new_val = corrupt_number(val_str, rng)
                corrupted = corrupted[:start] + new_val + corrupted[end:]
                corruption_log.append({"pos": start, "old": val_str, "new": new_val})
            new_ex["resp_after"] = corrupted
            new_ex["corruption_log"] = corruption_log
            total_corrupted += len(chosen)
        else:
            new_ex["resp_after"] = dense_text
            new_ex["corruption_log"] = []

        new_ex["resp_before"] = ex["resp_before"]
        new_ex["resp_rewrite_style"] = "e2_dense_incorrect"
        new_ex["resp_rewrite_ok"] = True
        results.append(new_ex)

    print(f"E2: Corrupted {total_corrupted} numbers across {len(results)} samples")
    return results


# ─── E3: Rule-Based Rewriting ───────────────────────────────

def bigram_overlap(s1: str, s2: str) -> float:
    def bigrams(s):
        tokens = s.lower().split()
        return Counter(zip(tokens, tokens[1:]))
    b1, b2 = bigrams(s1), bigrams(s2)
    if not b2:
        return 0.0
    overlap = sum((b1 & b2).values())
    return overlap / max(sum(b2.values()), 1)


def rule_based_dense_rewrite(steps: list) -> list:
    TRANSITION_PREFIXES = [
        "So,", "So ", "Thus,", "Thus ", "Therefore,", "Therefore ",
        "This means", "This gives", "That means", "Which means",
        "Now,", "Now ", "Next,", "Next ", "Let's", "Let us",
        "Moving on", "Alright,", "Okay,",
    ]
    SHORT_THRESHOLD = 20
    OVERLAP_THRESHOLD = 0.3

    def token_count(s):
        return len(s.split())

    def has_equation_marker(s):
        return "<<" in s and ">>" in s

    def starts_with_transition(s):
        return any(s.strip().startswith(p) for p in TRANSITION_PREFIXES)

    merged = []
    i = 0
    while i < len(steps):
        if i + 1 >= len(steps):
            merged.append(steps[i])
            i += 1
            break

        s_cur, s_next = steps[i], steps[i + 1]
        signals = 0

        cur_tail = " ".join(s_cur.split()[-15:])
        next_head = " ".join(s_next.split()[:15])
        if bigram_overlap(cur_tail, next_head) > OVERLAP_THRESHOLD:
            signals += 1

        if (token_count(s_cur) < SHORT_THRESHOLD
                and token_count(s_next) < SHORT_THRESHOLD
                and not has_equation_marker(s_cur)
                and not has_equation_marker(s_next)):
            signals += 1

        if starts_with_transition(s_next):
            signals += 1

        if signals >= 2:
            merged.append(s_cur.rstrip() + " " + s_next.lstrip())
            i += 2
        else:
            merged.append(s_cur)
            i += 1

    return merged


def prepare_e3(data, num_examples=50):
    em1 = [ex for ex in data if ex.get("exact_match", 0) == 1.0]
    selected = em1[-num_examples:]
    print(f"E3: {len(selected)} EM=1 samples selected for rule-based rewriting")

    results = []
    for ex in tqdm(selected, desc="E3 Rule-Based"):
        new_ex = deepcopy(ex)
        original = ex["resp_before"]
        steps = split_auto(original)
        merged_steps = rule_based_dense_rewrite(steps)

        separator = "\n\n" if "\n\n" in original else "\n"
        rewritten = separator.join(s.strip() for s in merged_steps)

        new_ex["resp_after"] = rewritten
        new_ex["resp_before"] = original
        new_ex["resp_rewrite_style"] = "e3_rule_based"
        new_ex["resp_rewrite_ok"] = True
        new_ex["steps_before"] = len(steps)
        new_ex["steps_after"] = len(merged_steps)
        results.append(new_ex)

    merged_cnt = sum(1 for r in results if r["steps_before"] > r["steps_after"])
    print(f"E3: {merged_cnt}/{len(results)} samples had steps merged")
    return results


# ─── E3.2: GPT-5-mini Rewriting ───────────────────────────

def build_prompt_old(question: str, original_resp: str) -> str:
    return f"""You will lightly rewrite the solution by CONSERVATIVELY merging steps, while keeping the SAME style and meaning.

Hard constraints:
- Keep the SAME meaning and do NOT change the final conclusion/answer implied by the solution.
- Do NOT invent new reasoning. Only compress/merge/rephrase existing steps.
- Keep the style and tone the SAME as the original (do not change formality, phrasing habits, or formatting conventions).
- Only merge steps when it is NECESSARY and safe (e.g., two adjacent lines that are clearly redundant or tightly coupled).
  Do NOT aggressively minimize the number of steps. If merging would change the "feel" or clarity, keep the original steps.
- When you merge, prefer merging 2 adjacent steps into 1 step (avoid merging many lines at once).
- Keep computations consistent with the original (same numbers/operations, no new math).
- Preserve special markers like "<<a=b>>" if they appear; do not introduce many new ones.
- Output plain text only. No bullet points or added commentary.

Question:
{question}

Original solution (model output):
{original_resp}

Now output ONLY the rewritten solution (same style, with a few necessary merges):
"""


def prepare_e3_2(data, client, model, num_examples=50):
    em1 = [ex for ex in data if ex.get("exact_match", 0) == 1.0]
    selected = em1[-num_examples:]
    print(f"E3.2: {len(selected)} EM=1 samples selected for {model} rewriting")

    results = []
    for ex in tqdm(selected, desc=f"E3.2 {model}"):
        new_ex = deepcopy(ex)
        original = ex["resp_before"]
        question = ex["doc"]["question"]
        prompt = build_prompt_old(question, original)
        rewritten = call_llm(client, model, prompt)
        new_ex["resp_after"] = rewritten
        new_ex["resp_before"] = original
        new_ex["resp_rewrite_style"] = "e3_2_gpt54mini"
        new_ex["resp_rewrite_ok"] = True
        results.append(new_ex)
    return results


# ─── Main ────────────────────────────────────────────────────

EXPERIMENT_TO_FILE = {
    "e0":   "rewritten_e0_paraphrase.json",
    "e1":   "rewritten_e1_random_step_compress.json",
    "e2":   "rewritten_e2_dense_incorrect.json",
    "e3":   "rewritten_e3_rule_based.json",
    "e3.2": "rewritten_e3_2_gpt54mini.json",
}


def main():
    parser = argparse.ArgumentParser(description="E0-E3 Ablation Data Preparation")
    parser.add_argument("--experiment", required=True,
                        choices=list(EXPERIMENT_TO_FILE.keys()),
                        help="Which experiment to prepare data for")
    parser.add_argument("--num_examples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", default=SOURCE_PATH,
                        help="Source data JSON (DenseSteer rewritten_old.json)")
    parser.add_argument("--api_model_e0_e1", default="gpt-5.1",
                        help="Model for E0/E1 API calls")
    parser.add_argument("--api_model_e3_2", default="gpt-5-mini",
                        help="Model for E3.2 API calls")
    args = parser.parse_args()

    exp = args.experiment
    out_path = os.path.join(OUTPUT_DIR, EXPERIMENT_TO_FILE[exp])
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Experiment: {exp}")
    print(f"Source:     {args.source}")
    print(f"Output:     {out_path}")

    data = json.load(open(args.source))
    print(f"Loaded {len(data)} samples ({sum(1 for x in data if x.get('exact_match',0)==1.0)} EM=1)")

    client = None
    if exp in EXPERIMENTS_NEEDING_API:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(BASE, ".env"))
        if not os.getenv("OPENAI_API_KEY"):
            print("ERROR: Set OPENAI_API_KEY for this experiment.", file=sys.stderr)
            sys.exit(2)
        from openai import OpenAI
        base_url = os.getenv("BASE_URL")
        client = OpenAI(base_url=base_url) if base_url else OpenAI()

    if exp == "e0":
        results = prepare_e0(data, client, args.api_model_e0_e1, args.num_examples)
    elif exp == "e1":
        results = prepare_e1(data, client, args.api_model_e0_e1, args.num_examples, args.seed)
    elif exp == "e2":
        results = prepare_e2(data, args.num_examples, args.seed)
    elif exp == "e3":
        results = prepare_e3(data, args.num_examples)
    elif exp == "e3.2":
        results = prepare_e3_2(data, client, args.api_model_e3_2, args.num_examples)
    else:
        raise ValueError(f"Unknown experiment: {exp}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} samples to {out_path}")


if __name__ == "__main__":
    main()
