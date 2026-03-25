#!/usr/bin/env python3
"""
Collect all rebuttal experiment results:
1. Control random compression experiment
2. Calibration size ablation (N=1,5,10,25,50)
3. Layer sensitivity (from existing sweeps)
4. Statistical significance (from previous run)
5. Open-source rewriting (when ready)

Outputs a summary table and JSON.
"""

import json
import os
import glob

BASE = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement"

def get_accuracy(results_dict, task="gsm8k_cot_zeroshot_unified"):
    task_results = results_dict.get(task, {})
    if "exact_match,flexible-extract" in task_results:
        return task_results["exact_match,flexible-extract"], task_results.get("exact_match_stderr,flexible-extract", 0)
    if "exact_match,none" in task_results:
        return task_results["exact_match,none"], task_results.get("exact_match_stderr,none", 0)
    return None, None

def find_results_json(directory):
    for f in glob.glob(os.path.join(directory, "**", "results_*.json"), recursive=True):
        return f
    return None

def collect_control_experiment():
    """Collect random compression control experiment results."""
    print("\n" + "="*60)
    print("1. CONTROL EXPERIMENT: Random Compression Steering")
    print("="*60)
    
    control_dir = os.path.join(BASE, "control_experiments", "Qwen_Qwen2.5-3B-Instruct")
    
    results = {}
    for d in glob.glob(os.path.join(control_dir, "eval_*")):
        tag = os.path.basename(d)
        rf = find_results_json(d)
        if rf:
            with open(rf) as f:
                data = json.load(f)
            acc, se = get_accuracy(data.get("results", {}))
            if acc is not None:
                results[tag] = {"accuracy": acc, "stderr": se}
                print(f"  {tag}: {acc*100:.2f}% (±{se*100:.2f}%)")
    
    if not results:
        print("  No results found yet.")
    return results

def collect_calibration_ablation():
    """Collect calibration size ablation results."""
    print("\n" + "="*60)
    print("2. CALIBRATION SIZE ABLATION")
    print("="*60)
    
    calib_dir = os.path.join(BASE, "calibration_ablation", "Qwen_Qwen2.5-3B-Instruct", "GPT_REWRITE")
    
    results = {}
    for N in [1, 5, 10, 25, 50]:
        eval_dir = os.path.join(calib_dir, f"eval_N{N}_L6_lam4.0")
        rf = find_results_json(eval_dir)
        if rf:
            with open(rf) as f:
                data = json.load(f)
            acc, se = get_accuracy(data.get("results", {}))
            if acc is not None:
                results[N] = {"accuracy": acc, "stderr": se}
                print(f"  N={N:3d}: {acc*100:.2f}% (±{se*100:.2f}%)")
    
    if not results:
        print("  No results found yet.")
    return results

def collect_baseline():
    """Find the zero-shot baseline accuracy for comparison."""
    print("\n" + "="*60)
    print("3. BASELINE COMPARISON")
    print("="*60)
    
    # Check existing steer_runs for baseline
    baseline_patterns = [
        os.path.join(BASE, "steer_runs*", "**", "gsm8k_cot_zeroshot_unified*", "**", "results_*.json"),
    ]
    
    # Also check existing compiled results
    compiled = os.path.join(BASE, "rebuttal_compiled_results.json")
    if os.path.exists(compiled):
        with open(compiled) as f:
            data = json.load(f)
        
        baselines = {}
        steered = {}
        for r in data.get("all_results", []):
            model = r.get("model", "")
            if "Qwen2.5-3B" in model and r.get("task") == "gsm8k_cot_zeroshot_unified":
                mode = r.get("steer_mode", "")
                acc = r.get("accuracy")
                se = r.get("stderr", 0)
                if mode == "Baseline" and acc is not None:
                    baselines[model] = {"accuracy": acc, "stderr": se}
                elif acc is not None and r.get("steer_layer"):
                    key = f"{mode}_L{r['steer_layer']}_lam{r['steer_lambda']}"
                    steered[key] = {"accuracy": acc, "stderr": se}
        
        for k, v in baselines.items():
            print(f"  Baseline ({k}): {v['accuracy']*100:.2f}%")
        for k, v in sorted(steered.items()):
            print(f"  Steered ({k}): {v['accuracy']*100:.2f}%")
        
        return baselines, steered
    
    print("  No compiled results found.")
    return {}, {}

def collect_layer_sensitivity():
    """Collect layer sensitivity data from existing sweeps."""
    print("\n" + "="*60)
    print("4. LAYER SENSITIVITY (from sweep data)")
    print("="*60)
    
    # Look for plot data
    plot_dir = os.path.join(BASE, "rebuttal_figures")
    if os.path.exists(plot_dir):
        for f in os.listdir(plot_dir):
            if f.endswith(".png"):
                print(f"  Plot: {f}")
    
    # Look for sweep results
    sweep_patterns = [
        os.path.join(BASE, "steer_runs*", "gsm8k*selected_layers*", "**", "results_*.json"),
    ]
    
    results = {}
    for pat in sweep_patterns:
        for rf in glob.glob(pat, recursive=True):
            try:
                with open(rf) as f:
                    data = json.load(f)
                acc, se = get_accuracy(data.get("results", {}))
                model_args = data.get("config", {}).get("model_args", "")
                layer = None
                lam = None
                for arg in model_args.split(","):
                    if "steer_layer=" in arg:
                        layer = int(arg.split("=")[1])
                    if "steer_lambda=" in arg:
                        lam = float(arg.split("=")[1])
                if acc is not None and layer is not None:
                    key = (layer, lam)
                    if key not in results or results[key]["accuracy"] < acc:
                        results[key] = {"accuracy": acc, "stderr": se, "layer": layer, "lambda": lam}
            except Exception:
                continue
    
    if results:
        print(f"  Found {len(results)} sweep configurations")
        best = max(results.values(), key=lambda x: x["accuracy"])
        print(f"  Best: layer={best['layer']}, lambda={best['lambda']}, acc={best['accuracy']*100:.2f}%")
    
    return results

def main():
    print("REBUTTAL EXPERIMENT RESULTS SUMMARY")
    print("Model: Qwen/Qwen2.5-3B-Instruct on GSM8K")
    print("Date: March 24, 2026")
    
    control = collect_control_experiment()
    calibration = collect_calibration_ablation()
    baselines, steered = collect_baseline()
    layer_data = collect_layer_sensitivity()
    
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    
    print(f"\n{'Experiment':<45} {'Accuracy':>10} {'Stderr':>10}")
    print("-" * 65)
    
    for k, v in baselines.items():
        print(f"{'Baseline (zero-shot)':<45} {v['accuracy']*100:>9.2f}% {v.get('stderr', 0)*100:>9.2f}%")
    
    for k, v in sorted(steered.items()):
        print(f"{'DenseSteer: ' + k:<45} {v['accuracy']*100:>9.2f}% {v.get('stderr', 0)*100:>9.2f}%")
    
    for k, v in control.items():
        print(f"{'Control: ' + k:<45} {v['accuracy']*100:>9.2f}% {v.get('stderr', 0)*100:>9.2f}%")
    
    for N in [1, 5, 10, 25, 50]:
        if N in calibration:
            v = calibration[N]
            print(f"{'Calibration N=' + str(N):<45} {v['accuracy']*100:>9.2f}% {v.get('stderr', 0)*100:>9.2f}%")
    
    # Save structured output
    output = {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "task": "gsm8k_cot_zeroshot_unified",
        "control_experiment": control,
        "calibration_ablation": {str(k): v for k, v in calibration.items()},
        "baselines": baselines,
        "steered": steered,
    }
    
    out_path = os.path.join(BASE, "rebuttal_experiment_summary.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
