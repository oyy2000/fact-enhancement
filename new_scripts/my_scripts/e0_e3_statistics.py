#!/usr/bin/env python3
"""
E0-E3 Statistics: Compute and display sample-level statistics for all ablation
experiments, comparing resp_before vs resp_after.

Metrics per sample:
  - N_steps (double-newline split)
  - total_tokens, tokens_per_step (mean)
  - char_length
  - reasoning density rho = tokens_per_step (higher = denser)

Aggregate: mean +/- std across EM=1 samples, printed as a table.

Also reports the original DenseSteer data for comparison.

Usage:
    python e0_e3_statistics.py
    python e0_e3_statistics.py --tokenizer Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import json
import os
import sys
import re
import numpy as np

BASE = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement"
CONTROL_DIR = os.path.join(BASE, "control_experiments", "Qwen_Qwen2.5-3B-Instruct")

DATA_FILES = {
    "DenseSteer (GPT-5.1)": os.path.join(
        BASE, "exps", "gpt_rewrites_unified_new",
        "Qwen_Qwen2.5-3B-Instruct", "rewritten_old.json",
    ),
    "E0: Paraphrase": os.path.join(CONTROL_DIR, "rewritten_e0_paraphrase.json"),
    "E1: Random Step Compress": os.path.join(CONTROL_DIR, "rewritten_e1_random_step_compress.json"),
    "E2: Dense Incorrect": os.path.join(CONTROL_DIR, "rewritten_e2_dense_incorrect.json"),
    "E3: Rule-Based": os.path.join(CONTROL_DIR, "rewritten_e3_rule_based.json"),
    "E3.2: GPT-5.4-mini": os.path.join(CONTROL_DIR, "rewritten_e3_2_gpt54mini.json"),
}


def split_double_newline(text):
    if not text:
        return []
    parts = text.split("\n\n")
    return [p for p in parts if p.strip()]


def compute_stats(texts, tokenizer):
    """Compute statistics for a list of text strings."""
    all_n_steps = []
    all_total_tokens = []
    all_tokens_per_step = []
    all_char_len = []

    for text in texts:
        if not text or not text.strip():
            continue
        steps = split_double_newline(text)
        n_steps = max(len(steps), 1)
        tokens_per_step_list = [
            len(tokenizer.encode(s, add_special_tokens=False)) for s in steps
        ] if steps else [0]
        total_tokens = sum(tokens_per_step_list)
        avg_tps = np.mean(tokens_per_step_list) if tokens_per_step_list else 0

        all_n_steps.append(n_steps)
        all_total_tokens.append(total_tokens)
        all_tokens_per_step.append(avg_tps)
        all_char_len.append(len(text))

    if not all_n_steps:
        return {k: (0, 0) for k in ["n_steps", "total_tokens", "tokens_per_step", "char_len"]}

    return {
        "n_steps": (np.mean(all_n_steps), np.std(all_n_steps)),
        "total_tokens": (np.mean(all_total_tokens), np.std(all_total_tokens)),
        "tokens_per_step": (np.mean(all_tokens_per_step), np.std(all_tokens_per_step)),
        "char_len": (np.mean(all_char_len), np.std(all_char_len)),
    }


def fmt(mean, std):
    return f"{mean:.1f} ± {std:.1f}"


def main():
    parser = argparse.ArgumentParser(description="E0-E3 Statistics")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--num_examples", type=int, default=50,
                        help="Max EM=1 samples to use (from the end)")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    header = (
        f"{'Experiment':<28} | {'Side':<6} | {'N_steps':>14} | "
        f"{'Total Tokens':>14} | {'Tok/Step (ρ)':>14} | {'Char Len':>14}"
    )
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for label, path in DATA_FILES.items():
        if not os.path.exists(path):
            print(f"{label:<28} | {'---':^6} | FILE NOT FOUND: {path}")
            continue

        data = json.load(open(path))
        em1 = [ex for ex in data if ex.get("exact_match", 0) == 1.0]
        selected = em1[-args.num_examples:]

        befores = [ex.get("resp_before", "") for ex in selected]
        afters = [ex.get("resp_after", "") for ex in selected]

        stats_before = compute_stats(befores, tokenizer)
        stats_after = compute_stats(afters, tokenizer)

        row_before = (
            f"{label:<28} | {'before':<6} | "
            f"{fmt(*stats_before['n_steps']):>14} | "
            f"{fmt(*stats_before['total_tokens']):>14} | "
            f"{fmt(*stats_before['tokens_per_step']):>14} | "
            f"{fmt(*stats_before['char_len']):>14}"
        )
        row_after = (
            f"{'':<28} | {'after':<6} | "
            f"{fmt(*stats_after['n_steps']):>14} | "
            f"{fmt(*stats_after['total_tokens']):>14} | "
            f"{fmt(*stats_after['tokens_per_step']):>14} | "
            f"{fmt(*stats_after['char_len']):>14}"
        )

        n_before = stats_before["n_steps"][0]
        n_after = stats_after["n_steps"][0]
        tps_before = stats_before["tokens_per_step"][0]
        tps_after = stats_after["tokens_per_step"][0]
        delta_steps = n_after - n_before
        delta_rho = tps_after - tps_before

        row_delta = (
            f"{'':<28} | {'Δ':<6} | "
            f"{delta_steps:>+14.1f} | "
            f"{'':>14} | "
            f"{delta_rho:>+14.1f} | "
            f"{'':>14}"
        )

        print(row_before)
        print(row_after)
        print(row_delta)
        print(sep)

    print(f"\nNote: ρ (tokens/step) = reasoning density. Higher means denser steps.")
    print(f"      Δ = after − before. Positive Δ(ρ) means steps became denser.")


if __name__ == "__main__":
    main()
