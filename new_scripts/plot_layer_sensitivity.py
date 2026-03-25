"""
Plot layer sensitivity analysis from existing GSM8K sweep data.
Addresses Reviewer jRak's request for ablation plots (Section 6.3).
"""
import json
import glob
import os
import re
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement"


def extract_gsm8k_sweep(model_dir, task="gsm8k_cot_zeroshot_unified"):
    """Extract all layer/lambda results from a model's sweep directory."""
    results = []
    
    for results_file in glob.glob(os.path.join(model_dir, "**/results_*.json"), recursive=True):
        try:
            with open(results_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            continue

        res = data.get("results", {}).get(task, {})
        acc_flex = res.get("exact_match,flexible-extract")
        acc_none = res.get("exact_match,none")
        acc = acc_flex if acc_flex is not None else acc_none
        if acc is None:
            continue

        config = data.get("configs", {}).get(task, {})
        metadata = config.get("metadata", {})
        layer = metadata.get("steer_layer")
        lam = metadata.get("steer_lambda")
        model_name = data.get("model_name", "")
        model_source = data.get("model_source", "")

        results.append({
            "model": model_name,
            "layer": layer,
            "lambda": lam,
            "accuracy": acc,
            "is_baseline": model_source != "steer_hf",
        })

    return results


def plot_layer_sensitivity_for_model(model_name, results, output_dir):
    """Plot accuracy vs layer for different lambda values, fixed model."""
    if not results:
        return

    steered = [r for r in results if not r["is_baseline"]]
    baselines = [r for r in results if r["is_baseline"]]
    baseline_acc = baselines[0]["accuracy"] if baselines else None

    by_layer = defaultdict(list)
    for r in steered:
        if r["layer"] is not None:
            by_layer[r["layer"]].append(r)

    layers = sorted(by_layer.keys())
    best_per_layer = []
    for l in layers:
        runs = by_layer[l]
        best = max(runs, key=lambda x: x["accuracy"])
        best_per_layer.append((l, best["accuracy"], best["lambda"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    
    l_vals = [x[0] for x in best_per_layer]
    acc_vals = [x[1] * 100 for x in best_per_layer]

    ax.plot(l_vals, acc_vals, 'o-', color='#2196F3', markersize=5, linewidth=1.5, label='Best DenseSteer')
    
    if baseline_acc is not None:
        ax.axhline(y=baseline_acc * 100, color='#F44336', linestyle='--',
                    linewidth=1.5, label=f'Baseline ({baseline_acc*100:.1f}%)')

    ax.set_xlabel("Injection Layer", fontsize=12)
    ax.set_ylabel("GSM8K Accuracy (%)", fontsize=12)
    short_model = model_name.split("/")[-1]
    ax.set_title(f"Layer Sensitivity: {short_model} (DenseSteer)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    path = os.path.join(output_dir, f"layer_sensitivity_{safe_name}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

    return best_per_layer


def plot_lambda_sensitivity_for_model(model_name, results, output_dir, target_layer=None):
    """Plot accuracy vs lambda for a specific layer (or best layer)."""
    steered = [r for r in results if not r["is_baseline"]]
    baselines = [r for r in results if r["is_baseline"]]
    baseline_acc = baselines[0]["accuracy"] if baselines else None

    if target_layer is None:
        by_layer = defaultdict(list)
        for r in steered:
            by_layer[r["layer"]].append(r)
        best_layer, best_acc = None, -1
        for l, runs in by_layer.items():
            layer_best = max(r["accuracy"] for r in runs)
            if layer_best > best_acc:
                best_acc = layer_best
                best_layer = l
        target_layer = best_layer

    layer_runs = [r for r in steered if r["layer"] == target_layer]
    if not layer_runs:
        return

    layer_runs.sort(key=lambda x: x["lambda"])
    lambdas = [r["lambda"] for r in layer_runs]
    accs = [r["accuracy"] * 100 for r in layer_runs]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lambdas, accs, 'o-', color='#4CAF50', markersize=5, linewidth=1.5)

    if baseline_acc is not None:
        ax.axhline(y=baseline_acc * 100, color='#F44336', linestyle='--',
                    linewidth=1.5, label=f'Baseline ({baseline_acc*100:.1f}%)')

    ax.set_xlabel(f"Steering Multiplier (λ) at Layer {target_layer}", fontsize=12)
    ax.set_ylabel("GSM8K Accuracy (%)", fontsize=12)
    short_model = model_name.split("/")[-1]
    ax.set_title(f"Lambda Sensitivity: {short_model} (Layer {target_layer})", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    path = os.path.join(output_dir, f"lambda_sensitivity_{safe_name}_L{target_layer}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    output_dir = os.path.join(BASE, "rebuttal_figures")

    sweep_dirs = {
        "DenseSteer": glob.glob(os.path.join(BASE,
            "gpt_rewrites_unified_new/*/vectors_50_*/*/gsm8k_cot_zeroshot_unified_selected_layers*")),
        "InFamilySteer": glob.glob(os.path.join(BASE,
            "large_model_rewrites_unified_new/*/vectors_50_*/*/gsm8k_cot_zeroshot_unified_selected_layers*")),
    }

    for mode, dirs in sweep_dirs.items():
        print(f"\n{'='*60}")
        print(f"Mode: {mode}")
        print(f"{'='*60}")

        all_results = []
        for d in dirs:
            results = extract_gsm8k_sweep(d)
            all_results.extend(results)

        by_model = defaultdict(list)
        for r in all_results:
            by_model[r["model"]].append(r)

        for model_name in sorted(by_model.keys()):
            model_results = by_model[model_name]
            print(f"\n--- {model_name} ({len(model_results)} runs) ---")
            
            best_per_layer = plot_layer_sensitivity_for_model(
                model_name, model_results,
                os.path.join(output_dir, mode))

            plot_lambda_sensitivity_for_model(
                model_name, model_results,
                os.path.join(output_dir, mode))

            if best_per_layer:
                top5 = sorted(best_per_layer, key=lambda x: -x[1])[:5]
                for l, acc, lam in top5:
                    print(f"    Layer {l:3d}: {acc*100:5.1f}% (λ={lam})")


if __name__ == "__main__":
    main()
