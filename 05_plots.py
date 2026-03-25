#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer


ROOT = Path(
    "/common/users/sl2148/Public/yang_ouyang/projects/"
    "fact-enhancement/no_vector/gsm8k_cot_zeroshot_unified"
)

# 只处理 Qwen
MODEL_PREFIX = "Qwen2.5"

# 哪些 filter 的样本参与统计（之前只看 flexible-match 导致无数据）
ACCEPT_FILTERS = {"flexible-extract"}

def find_qwen_sample_files(root: Path):
    files = []
    for p in root.rglob("samples_*.jsonl"):
        if MODEL_PREFIX in str(p):
            files.append(p)
    return files


def load_samples(path: Path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def get_model_name_from_path(path: Path):
    """
    e.g.
    Qwen2.5-0.5B-Instruct_no_vector/Qwen__Qwen2.5-0.5B-Instruct/samples_*.jsonl
    -> Qwen/Qwen2.5-0.5B-Instruct
    """
    for part in path.parts:
        if part.startswith("Qwen__"):
            return part.replace("Qwen__", "Qwen/")
    raise ValueError(f"Cannot infer model name from {path}")


def extract_first_text(obj):
    """Return the first response text from several possible fields."""
    for key in ("resps", "filtered_resps"):
        cand = obj.get(key)
        if not cand:
            continue
        first = cand[0]
        # Some entries are nested like [["text"]]
        while isinstance(first, list) and first:
            first = first[0]
        if isinstance(first, str):
            return first
    return None


def get_exact_match_flag(obj):
    """Return 1/0 for correctness, handling different exact_match keys and types."""
    val = obj.get("exact-match")
    if val is None:
        val = obj.get("exact_match")
    if val is None:
        return 0
    try:
        v = float(val)
    except (TypeError, ValueError):
        return 0
    return 1 if v >= 0.5 else 0


def main():
    sample_files = find_qwen_sample_files(ROOT)
    print(f"[Found] {len(sample_files)} Qwen sample files")

    results = {}

    for sample_path in sample_files:
        model_name = get_model_name_from_path(sample_path)
        print(f"\nProcessing {model_name}")
        print(f"  File: {sample_path}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        tokens_per_step_list = []
        steps_counts = []
        acc_flags = []

        for obj in load_samples(sample_path):
            if obj.get("filter") not in ACCEPT_FILTERS:
                continue

            resp = extract_first_text(obj)
            if not resp:
                continue

            # 按 \n\n 划分 step
            steps = [s.strip() for s in resp.split("\n\n") if s.strip()]
            if not steps:
                continue

            step_token_lens = []
            for step in steps:
                toks = tokenizer.encode(step, add_special_tokens=False)
                step_token_lens.append(len(toks))

            tokens_per_step_list.append(float(np.mean(step_token_lens)))
            steps_counts.append(len(steps))

            acc_flags.append(get_exact_match_flag(obj))

        if len(acc_flags) == 0:
            print("  [Skip] no accepted samples")
            continue

        avg_tokens_per_step = float(np.mean(tokens_per_step_list))
        avg_steps = float(np.mean(steps_counts))
        acc = float(np.mean(acc_flags))

        results[model_name] = {
            "avg_tokens_per_step": avg_tokens_per_step,
            "avg_steps": avg_steps,
            "accuracy": acc,
            "n": len(acc_flags),
        }

        print(f"  Avg tokens/step: {avg_tokens_per_step:.2f}")
        print(f"  Avg steps     : {avg_steps:.2f}")
        print(f"  Accuracy      : {acc:.3f} ({len(acc_flags)} samples)")

    if not results:
        print("[Exit] no flexible-match samples found; nothing to plot.")
        return

    # ================= Plot 1: tokens per step =================
    plt.figure(figsize=(7, 5))

    for model, stats in results.items():
        plt.scatter(
            stats["avg_tokens_per_step"],
            stats["accuracy"],
            label=model.split("/")[-1],
            s=80,
        )

    plt.xlabel("Avg Tokens per Step")
    plt.ylabel("Accuracy")
    plt.title("Qwen Models: Avg Tokens/Step vs Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("qwen_tokens_per_step_vs_accuracy.png", dpi=200)

    # ================= Plot 2: number of steps =================
    plt.figure(figsize=(7, 5))

    for model, stats in results.items():
        plt.scatter(
            stats["avg_steps"],
            stats["accuracy"],
            label=model.split("/")[-1],
            s=80,
        )

    plt.xlabel("Avg #Steps per Response")
    plt.ylabel("Accuracy")
    plt.title("Qwen Models: Steps vs Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("qwen_steps_vs_accuracy.png", dpi=200)

    plt.show()


if __name__ == "__main__":
    main()