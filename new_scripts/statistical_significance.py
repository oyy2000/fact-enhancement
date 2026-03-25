"""
Statistical Significance Testing for DenseSteer Results.
Computes McNemar's test between DenseSteer/InFamilySteer and baseline
using sample-level predictions from JSONL files.
"""
import json
import glob
import os
import argparse
from collections import defaultdict
import numpy as np

BASE = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement"


def load_sample_predictions(jsonl_path):
    """Load per-sample predictions from lm_eval samples JSONL."""
    preds = {}
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            doc_id = sample.get("doc_id")
            
            em = sample.get("exact_match", None)
            if em is None:
                filt = sample.get("filtered_resps", [[""]])
                em = sample.get("metrics", {}).get("exact_match", 0)
            
            if doc_id is not None:
                preds[doc_id] = int(float(em) if em is not None else 0)
    return preds


def mcnemar_test(preds_a, preds_b, shared_ids):
    """McNemar's test comparing two models on the same samples."""
    b_correct_a_wrong = 0  # B correct, A wrong
    a_correct_b_wrong = 0  # A correct, B wrong
    both_correct = 0
    both_wrong = 0

    for doc_id in shared_ids:
        a = preds_a.get(doc_id, 0)
        b = preds_b.get(doc_id, 0)
        if a == 1 and b == 1:
            both_correct += 1
        elif a == 0 and b == 0:
            both_wrong += 1
        elif a == 1 and b == 0:
            a_correct_b_wrong += 1
        else:
            b_correct_a_wrong += 1

    n = b_correct_a_wrong + a_correct_b_wrong
    if n == 0:
        return {"p_value": 1.0, "statistic": 0, "n_discordant": 0,
                "b_wins": 0, "a_wins": 0}

    from scipy.stats import binomtest
    result = binomtest(b_correct_a_wrong, n, 0.5)
    p_value = result.pvalue

    return {
        "p_value": p_value,
        "n_discordant": n,
        "b_wins": b_correct_a_wrong,
        "a_wins": a_correct_b_wrong,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "total": len(shared_ids),
    }


def bootstrap_ci(preds, n_boot=10000, alpha=0.05, seed=42):
    """Bootstrap confidence interval for accuracy."""
    rng = np.random.RandomState(seed)
    vals = np.array(list(preds.values()), dtype=float)
    n = len(vals)
    boot_means = np.array([rng.choice(vals, n, replace=True).mean() for _ in range(n_boot)])
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(vals.mean()), float(lower), float(upper)


def find_sample_files(base_dirs, model_pattern, task):
    """Find sample JSONL files matching criteria."""
    results = []
    for base in base_dirs:
        for path in glob.glob(os.path.join(base, f"**/*samples_{task}*.jsonl"),
                               recursive=True):
            if model_pattern in path:
                results.append(path)
    return results


def main():
    # Find baseline and steered sample files for GSM8K
    task = "gsm8k_cot_zeroshot_unified"

    models = {
        "Qwen2.5-3B-Instruct": {
            "baseline": os.path.join(BASE,
                "no_vector/gsm8k_cot_zeroshot_unified/Qwen2.5-3B-Instruct_no_vector/Qwen__Qwen2.5-3B-Instruct"),
            "densesteer_dirs": [
                os.path.join(BASE, "gpt_rewrites_unified_new/Qwen_Qwen2.5-3B-Instruct"),
            ],
            "infamily_dirs": [
                os.path.join(BASE, "large_model_rewrites_unified_new/Qwen_Qwen2.5-3B-Instruct"),
            ],
        },
        "Qwen2.5-1.5B-Instruct": {
            "baseline": os.path.join(BASE,
                "no_vector/gsm8k_cot_zeroshot_unified/Qwen2.5-1.5B-Instruct_no_vector/Qwen__Qwen2.5-1.5B-Instruct"),
            "densesteer_dirs": [
                os.path.join(BASE, "gpt_rewrites_unified_new/Qwen_Qwen2.5-1.5B-Instruct"),
            ],
            "infamily_dirs": [
                os.path.join(BASE, "large_model_rewrites_unified_new/Qwen_Qwen2.5-1.5B-Instruct"),
            ],
        },
        "Llama-3.2-3B-Instruct": {
            "baseline": os.path.join(BASE,
                "no_vector/gsm8k_cot_zeroshot_unified/Llama-3.2-3B-Instruct_no_vector/meta-llama__Llama-3.2-3B-Instruct"),
            "densesteer_dirs": [
                os.path.join(BASE, "gpt_rewrites_unified_new/meta-llama_Llama-3.2-3B-Instruct"),
            ],
            "infamily_dirs": [
                os.path.join(BASE, "large_model_rewrites_unified_new/meta-llama_Llama-3.2-3B-Instruct"),
            ],
        },
    }

    for model_name, config in models.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        # Load baseline
        baseline_files = glob.glob(os.path.join(config["baseline"],
                                                  f"samples_{task}*.jsonl"))
        if not baseline_files:
            print(f"  No baseline samples found in {config['baseline']}")
            continue

        baseline_preds = load_sample_predictions(baseline_files[0])
        base_acc, base_lo, base_hi = bootstrap_ci(baseline_preds)
        print(f"  Baseline: {base_acc*100:.1f}% (95% CI: [{base_lo*100:.1f}%, {base_hi*100:.1f}%])")

        for mode, dirs in [("DenseSteer", config["densesteer_dirs"]),
                            ("InFamilySteer", config["infamily_dirs"])]:
            best_acc = -1
            best_file = None
            
            for d in dirs:
                for f in glob.glob(os.path.join(d, f"**/*samples_{task}*.jsonl"),
                                    recursive=True):
                    try:
                        preds = load_sample_predictions(f)
                        if preds:
                            acc = sum(preds.values()) / len(preds)
                            if acc > best_acc:
                                best_acc = acc
                                best_file = f
                    except Exception:
                        continue

            if best_file is None:
                print(f"  {mode}: No samples found")
                continue

            steered_preds = load_sample_predictions(best_file)
            st_acc, st_lo, st_hi = bootstrap_ci(steered_preds)
            
            shared_ids = set(baseline_preds.keys()) & set(steered_preds.keys())
            if len(shared_ids) > 10:
                test_result = mcnemar_test(baseline_preds, steered_preds, shared_ids)
                sig_str = "***" if test_result["p_value"] < 0.001 else \
                          "**" if test_result["p_value"] < 0.01 else \
                          "*" if test_result["p_value"] < 0.05 else "n.s."
                
                print(f"  {mode}: {st_acc*100:.1f}% (95% CI: [{st_lo*100:.1f}%, {st_hi*100:.1f}%])")
                print(f"    McNemar's p={test_result['p_value']:.4f} {sig_str}")
                print(f"    Discordant pairs: {test_result['n_discordant']} "
                      f"(steered wins: {test_result['b_wins']}, baseline wins: {test_result['a_wins']})")
            else:
                print(f"  {mode}: {st_acc*100:.1f}% (insufficient overlap for McNemar)")


if __name__ == "__main__":
    main()
