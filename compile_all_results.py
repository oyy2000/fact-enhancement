"""
Compile all existing experiment results into a comprehensive rebuttal table.
Handles:
- no_vector/ baselines (GSM8K + MATH-500/AIME/AMC multi-task)
- steer_runs*/ steered results (GPT_REWRITE + LARGE_MODEL)
- long_cot_vs_short_cot*/ KD baselines
- gpt_rewrites_unified_new/*/gsm8k sweeps
- large_model_rewrites_unified_new/*/gsm8k sweeps
"""
import json
import glob
import os
from collections import defaultdict

BASE = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement"


def get_accuracy(results_dict, task):
    """Extract accuracy from a results dict for a specific task."""
    task_results = results_dict.get(task, {})
    if "exact_match,flexible-extract" in task_results:
        return task_results["exact_match,flexible-extract"], task_results.get("exact_match_stderr,flexible-extract", 0)
    if "exact_match,none" in task_results:
        return task_results["exact_match,none"], task_results.get("exact_match_stderr,none", 0)
    return None, None


def extract_all_tasks(path):
    """Extract results for ALL tasks from a results JSON file."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

    results_dict = data.get("results", {})
    if not results_dict:
        return []

    model_name = data.get("model_name", "")
    model_source = data.get("model_source", "")

    entries = []
    for task in results_dict:
        acc, stderr = get_accuracy(results_dict, task)
        if acc is None:
            continue

        config = data.get("configs", {}).get(task, {})
        metadata = config.get("metadata", {})
        steer_layer = metadata.get("steer_layer", None)
        steer_lambda = metadata.get("steer_lambda", None)
        steer_vec_path = metadata.get("steer_vec_path", "")

        if "gpt_rewrites" in steer_vec_path:
            steer_mode = "DenseSteer"
        elif "large_model_rewrites" in steer_vec_path:
            steer_mode = "InFamilySteer"
        elif model_source == "steer_hf":
            steer_mode = "Steer_Unknown"
        else:
            steer_mode = "Baseline"

        entries.append({
            "path": path,
            "model": model_name,
            "task": task,
            "accuracy": acc,
            "stderr": stderr,
            "model_source": model_source,
            "steer_mode": steer_mode,
            "steer_layer": steer_layer,
            "steer_lambda": steer_lambda,
        })

    return entries


def scan_all():
    all_results = []
    seen_paths = set()

    search_dirs = [
        os.path.join(BASE, "no_vector"),
        os.path.join(BASE, "steer_runs"),
        os.path.join(BASE, "steer_runs_2"),
        os.path.join(BASE, "steer_runs_3"),
        os.path.join(BASE, "steer_runs_4"),
        os.path.join(BASE, "steer_runs_5"),
        os.path.join(BASE, "steer_runs_6"),
        os.path.join(BASE, "long_cot_vs_short_cot"),
        os.path.join(BASE, "long_cot_vs_short_cot_2"),
        os.path.join(BASE, "long_cot_vs_short_cot_dense_2"),
        os.path.join(BASE, "long_cot_vs_short_cot_dense_4"),
        os.path.join(BASE, "long_cot_vs_short_cot_5"),
        os.path.join(BASE, "gpt_rewrites_unified_new"),
        os.path.join(BASE, "large_model_rewrites_unified_new"),
    ]

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for path in glob.glob(os.path.join(d, "**/results_*.json"), recursive=True):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            entries = extract_all_tasks(path)
            all_results.extend(entries)

    return all_results


def find_best(results, model, task, mode=None):
    filtered = [r for r in results if r["model"] == model and r["task"] == task]
    if mode:
        filtered = [r for r in filtered if r["steer_mode"] == mode]
    if not filtered:
        return None
    return max(filtered, key=lambda x: x["accuracy"])


def main():
    results = scan_all()
    print(f"Total result entries: {len(results)}\n")

    # Normalize task name for GSM8K
    task_map = {
        "gsm8k_cot_zeroshot_unified": "GSM8K",
        "hendrycks_math_500": "MATH-500",
        "AIME": "AIME",
        "AMC": "AMC",
        "Olympiad": "Olympiad",
    }

    # ====== SECTION 1: Baselines for all models ======
    print("=" * 120)
    print("SECTION 1: NO-STEER BASELINES (from no_vector/)")
    print("=" * 120)

    baseline_models = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
    ]
    tasks_core = ["gsm8k_cot_zeroshot_unified", "hendrycks_math_500", "AIME", "AMC", "Olympiad"]
    baselines = [r for r in results if r["steer_mode"] == "Baseline"]

    header = f"{'Model':42s}"
    for t in tasks_core:
        header += f" | {task_map.get(t, t):12s}"
    print(header)
    print("-" * len(header))

    for model in baseline_models:
        short = model.split("/")[-1]
        row = f"{short:42s}"
        for task in tasks_core:
            best = find_best(baselines, model, task)
            if best:
                row += f" | {best['accuracy']*100:5.1f} ±{best['stderr']*100:4.1f}"
            else:
                row += f" | {'N/A':>12s}"
        print(row)

    # ====== SECTION 2: KD Baselines ======
    print("\n" + "=" * 120)
    print("SECTION 2: KNOWLEDGE DISTILLATION BASELINES (from long_cot_vs_short_cot/)")
    print("=" * 120)

    kd_models = sorted(set(r["model"] for r in baselines if "UWNSL" in r["model"]))
    tasks_kd = ["hendrycks_math_500", "AIME", "AMC", "Olympiad"]

    header = f"{'KD Model':50s}"
    for t in tasks_kd:
        header += f" | {task_map.get(t, t):12s}"
    print(header)
    print("-" * len(header))

    for model in kd_models:
        short = model.split("/")[-1]
        row = f"{short:50s}"
        for task in tasks_kd:
            best = find_best(baselines, model, task)
            if best:
                row += f" | {best['accuracy']*100:5.1f} ±{best['stderr']*100:4.1f}"
            else:
                row += f" | {'N/A':>12s}"
        print(row)

    # ====== SECTION 3: DenseSteer & InFamilySteer best per model/task ======
    print("\n" + "=" * 120)
    print("SECTION 3: STEERING RESULTS (best per model/task)")
    print("=" * 120)

    steered = [r for r in results if r["steer_mode"] in ("DenseSteer", "InFamilySteer")]
    target_models = [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
    ]
    tasks_all = ["gsm8k_cot_zeroshot_unified", "hendrycks_math_500", "AIME", "AMC", "Olympiad"]

    for model in target_models:
        short = model.split("/")[-1]
        print(f"\n--- {short} ---")
        header = f"  {'Method':15s}"
        for t in tasks_all:
            header += f" | {task_map.get(t, t):14s}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        # Baseline
        row = f"  {'Baseline':15s}"
        for task in tasks_all:
            best = find_best(baselines, model, task)
            if best:
                row += f" | {best['accuracy']*100:5.1f} ±{best['stderr']*100:4.1f} "
            else:
                row += f" | {'N/A':>14s}"
        print(row)

        for mode in ["DenseSteer", "InFamilySteer"]:
            row = f"  {mode:15s}"
            for task in tasks_all:
                best = find_best(steered, model, task, mode)
                if best:
                    row += f" | {best['accuracy']*100:5.1f} ±{best['stderr']*100:4.1f} "
                else:
                    row += f" | {'N/A':>14s}"
            print(row)

    # ====== SECTION 4: GSM8K sweep summary ======
    print("\n" + "=" * 120)
    print("SECTION 4: GSM8K LAYER/LAMBDA SWEEP SUMMARY (best configs)")
    print("=" * 120)

    gsm8k_steered = [r for r in steered if r["task"] == "gsm8k_cot_zeroshot_unified"]
    for model in target_models:
        short = model.split("/")[-1]
        for mode in ["DenseSteer", "InFamilySteer"]:
            filtered = [r for r in gsm8k_steered if r["model"] == model and r["steer_mode"] == mode]
            if not filtered:
                continue
            best = max(filtered, key=lambda x: x["accuracy"])
            print(f"  {short:35s} | {mode:15s} | acc={best['accuracy']*100:5.1f}% | L={best['steer_layer']}, λ={best['steer_lambda']}")
            # Also show top-5
            top5 = sorted(filtered, key=lambda x: -x["accuracy"])[:5]
            for r in top5:
                print(f"    L={r['steer_layer']:3s}, λ={str(r['steer_lambda']):6s} -> {r['accuracy']*100:5.1f}%"
                      if isinstance(r['steer_layer'], str) else
                      f"    L={r['steer_layer']:3d}, λ={r['steer_lambda']:6.1f} -> {r['accuracy']*100:5.1f}%")

    # ====== SECTION 5: Dense prompt baselines ======
    print("\n" + "=" * 120)
    print("SECTION 5: DENSE PROMPT BASELINES (no steering, dense task variants)")
    print("=" * 120)

    dense_tasks = ["hendrycks_math_500_dense", "AIME_dense", "AMC_dense", "Olympiad_dense"]
    for model in target_models:
        short = model.split("/")[-1]
        row = f"  {short:42s}"
        for task in dense_tasks:
            best = find_best(baselines, model, task)
            if best:
                row += f" | {task.replace('_dense','')}: {best['accuracy']*100:5.1f}%"
            else:
                row += f" | {task.replace('_dense','')}: N/A"
        print(row)

    # Export JSON
    out_path = os.path.join(BASE, "rebuttal_compiled_results.json")
    output = {"all_results": []}
    for r in results:
        output["all_results"].append({
            "model": r["model"], "task": r["task"],
            "accuracy": r["accuracy"], "stderr": r["stderr"],
            "steer_mode": r["steer_mode"],
            "steer_layer": r["steer_layer"], "steer_lambda": r["steer_lambda"],
        })
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nExported {len(output['all_results'])} entries to {out_path}")


if __name__ == "__main__":
    main()
