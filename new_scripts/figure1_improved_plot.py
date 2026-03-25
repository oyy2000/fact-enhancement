#!/usr/bin/env python3
"""
Improved Figure 1: Reasoning Density vs Accuracy for the Qwen-2.5 family.

Generates multiple panels from sampling data:
  (A) Model-level: avg ρ vs accuracy with error bars + regression (classic Fig 1)
  (B) Within-model: density distribution of correct vs incorrect, per model
  (C) Sample-level logistic regression: P(correct) ~ ρ, pooled across models
  (D) Within-question: for each question, compare density of correct vs incorrect

Usage:
    python figure1_improved_plot.py [--data_dir figure1_sampling_data]
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUT_DIR = Path("/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/new_exps/figures/figure1_improved")

MODEL_ORDER = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
]

MODEL_SHORT = {
    "Qwen/Qwen2.5-0.5B-Instruct": "0.5B",
    "Qwen/Qwen2.5-1.5B-Instruct": "1.5B",
    "Qwen/Qwen2.5-3B-Instruct": "3B",
    "Qwen/Qwen2.5-7B-Instruct": "7B",
    "Qwen/Qwen2.5-14B-Instruct": "14B",
    "Qwen/Qwen2.5-32B-Instruct": "32B",
    "Qwen/Qwen2.5-72B-Instruct": "72B",
}

MODEL_SIZE_B = {
    "Qwen/Qwen2.5-0.5B-Instruct": 0.5,
    "Qwen/Qwen2.5-1.5B-Instruct": 1.5,
    "Qwen/Qwen2.5-3B-Instruct": 3.0,
    "Qwen/Qwen2.5-7B-Instruct": 7.0,
    "Qwen/Qwen2.5-14B-Instruct": 14.0,
    "Qwen/Qwen2.5-32B-Instruct": 32.0,
    "Qwen/Qwen2.5-72B-Instruct": 72.0,
}


def load_sampling_data(data_dir: Path):
    """Load all per-sample data from sampling JSONL files."""
    all_data = {}  # model -> list of {doc_id, question, samples: [...]}

    for model in MODEL_ORDER:
        sanitized = model.replace("/", "_")
        fpath = data_dir / sanitized / "gsm8k_samples.jsonl"
        if not fpath.exists():
            print(f"  [skip] {fpath} not found")
            continue

        records = []
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        all_data[model] = records
        n_q = len(records)
        n_s = sum(len(r["samples"]) for r in records)
        print(f"  Loaded {model}: {n_q} questions, {n_s} samples")

    return all_data


def bootstrap_ci(values, n_boot=2000, ci=0.95):
    values = np.array(values)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    boot_means = np.array([
        np.mean(np.random.choice(values, size=n, replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, 100 * alpha)
    hi = np.percentile(boot_means, 100 * (1 - alpha))
    return np.mean(values), lo, hi


def plot_panel_A(all_data, ax):
    """Model-level: avg density (ρ) vs accuracy with error bars and regression."""
    xs, ys, x_lo, x_hi, y_lo, y_hi, labels = [], [], [], [], [], [], []

    for model in MODEL_ORDER:
        if model not in all_data:
            continue
        records = all_data[model]

        all_densities = []
        all_correct = []
        for rec in records:
            for s in rec["samples"]:
                all_densities.append(s["density_rho"])
                all_correct.append(1.0 if s["correct"] else 0.0)

        d_mean, d_lo, d_hi = bootstrap_ci(all_densities)
        a_mean, a_lo, a_hi = bootstrap_ci(all_correct)

        xs.append(d_mean)
        ys.append(a_mean)
        x_lo.append(d_mean - d_lo)
        x_hi.append(d_hi - d_mean)
        y_lo.append(a_mean - a_lo)
        y_hi.append(a_hi - a_mean)
        labels.append(MODEL_SHORT.get(model, model.split("/")[-1]))

    xs, ys = np.array(xs), np.array(ys)

    ax.errorbar(xs, ys,
                xerr=[x_lo, x_hi], yerr=[y_lo, y_hi],
                fmt="o", markersize=10, capsize=4, capthick=1.5,
                color="#2563EB", ecolor="#93C5FD", zorder=5)

    for i, label in enumerate(labels):
        ax.annotate(label, (xs[i], ys[i]),
                    textcoords="offset points", xytext=(8, 6),
                    fontsize=11, fontweight="bold")

    if len(xs) >= 3:
        slope, intercept, r_value, p_value, _ = stats.linregress(xs, ys)
        x_fit = np.linspace(xs.min() - 2, xs.max() + 2, 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, "--", color="#DC2626", alpha=0.7, linewidth=1.5)
        rho, p_spearman = stats.spearmanr(xs, ys)
        ax.text(0.05, 0.05,
                f"Spearman ρ = {rho:.2f} (p = {p_spearman:.3f})\n"
                f"Pearson r = {r_value:.2f} (p = {p_value:.3f})",
                transform=ax.transAxes, fontsize=9,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Avg Reasoning Density ρ (tokens/step)")
    ax.set_ylabel("Accuracy")
    ax.set_title("(a) Model-Level: Density vs Accuracy", fontweight="bold")
    ax.grid(alpha=0.3)


def plot_panel_B(all_data, ax):
    """Within-model: density of correct vs incorrect responses, violin plot."""
    positions = []
    violin_data_correct = []
    violin_data_incorrect = []
    tick_labels = []

    for i, model in enumerate(MODEL_ORDER):
        if model not in all_data:
            continue
        records = all_data[model]

        correct_densities = []
        incorrect_densities = []
        for rec in records:
            for s in rec["samples"]:
                if s["correct"]:
                    correct_densities.append(s["density_rho"])
                else:
                    incorrect_densities.append(s["density_rho"])

        if not correct_densities or not incorrect_densities:
            continue

        positions.append(i)
        violin_data_correct.append(correct_densities)
        violin_data_incorrect.append(incorrect_densities)
        short = MODEL_SHORT.get(model, model.split("/")[-1])
        acc = len(correct_densities) / (len(correct_densities) + len(incorrect_densities))
        tick_labels.append(f"{short}\n({acc:.1%})")

    if not positions:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return

    width = 0.35
    bp_correct = ax.boxplot(
        violin_data_correct,
        positions=[p - width / 2 for p in positions],
        widths=width * 0.8,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="#BFDBFE", edgecolor="#2563EB"),
        medianprops=dict(color="#1E40AF", linewidth=2),
    )
    bp_incorrect = ax.boxplot(
        violin_data_incorrect,
        positions=[p + width / 2 for p in positions],
        widths=width * 0.8,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="#FECACA", edgecolor="#DC2626"),
        medianprops=dict(color="#991B1B", linewidth=2),
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_ylabel("Reasoning Density ρ (tokens/step)")
    ax.set_title("(b) Within-Model: Correct vs Incorrect", fontweight="bold")
    ax.legend(
        [bp_correct["boxes"][0], bp_incorrect["boxes"][0]],
        ["Correct", "Incorrect"],
        loc="upper right", fontsize=9,
    )
    ax.grid(axis="y", alpha=0.3)


def plot_panel_C(all_data, ax):
    """Sample-level: binned P(correct) vs density, per model."""
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(MODEL_ORDER)))

    for idx, model in enumerate(MODEL_ORDER):
        if model not in all_data:
            continue
        records = all_data[model]

        densities = []
        corrects = []
        for rec in records:
            for s in rec["samples"]:
                densities.append(s["density_rho"])
                corrects.append(1.0 if s["correct"] else 0.0)

        densities = np.array(densities)
        corrects = np.array(corrects)

        n_bins = 10
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(densities, percentiles)

        bin_centers = []
        bin_accs = []
        bin_sems = []

        for j in range(n_bins):
            lo, hi = bin_edges[j], bin_edges[j + 1]
            if j < n_bins - 1:
                mask = (densities >= lo) & (densities < hi)
            else:
                mask = (densities >= lo) & (densities <= hi)

            if mask.sum() < 5:
                continue
            bin_centers.append(np.mean(densities[mask]))
            acc = np.mean(corrects[mask])
            sem = np.std(corrects[mask]) / np.sqrt(mask.sum())
            bin_accs.append(acc)
            bin_sems.append(sem)

        short = MODEL_SHORT.get(model, model.split("/")[-1])
        ax.errorbar(bin_centers, bin_accs, yerr=bin_sems,
                    fmt="-o", markersize=4, capsize=2, linewidth=1.5,
                    color=colors[idx], label=short, alpha=0.85)

    ax.set_xlabel("Reasoning Density ρ (tokens/step)")
    ax.set_ylabel("P(correct)")
    ax.set_title("(c) Sample-Level: Accuracy by Density Bin", fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)


def plot_panel_D(all_data, ax):
    """Within-question analysis: for questions with both correct and incorrect 
    samples, compare their density distributions."""
    delta_densities = defaultdict(list)  # model -> list of (correct_avg - incorrect_avg)

    for model in MODEL_ORDER:
        if model not in all_data:
            continue
        for rec in all_data[model]:
            correct_d = [s["density_rho"] for s in rec["samples"] if s["correct"]]
            incorrect_d = [s["density_rho"] for s in rec["samples"] if not s["correct"]]
            if correct_d and incorrect_d:
                delta = np.mean(correct_d) - np.mean(incorrect_d)
                delta_densities[model].append(delta)

    models_with_data = [m for m in MODEL_ORDER if m in delta_densities and delta_densities[m]]
    if not models_with_data:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return

    positions = list(range(len(models_with_data)))
    data_list = [delta_densities[m] for m in models_with_data]
    tick_labels = [MODEL_SHORT.get(m, m.split("/")[-1]) for m in models_with_data]

    bp = ax.boxplot(data_list, positions=positions, widths=0.5,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor="#E0E7FF", edgecolor="#4F46E5"),
                    medianprops=dict(color="#312E81", linewidth=2))

    for i, (pos, deltas) in enumerate(zip(positions, data_list)):
        mean_val = np.mean(deltas)
        ax.plot(pos, mean_val, "D", color="#DC2626", markersize=8, zorder=5)
        t_stat, p_val = stats.ttest_1samp(deltas, 0)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        ax.text(pos, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 5,
                sig, ha="center", fontsize=11, fontweight="bold")

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Δρ (correct − incorrect)")
    ax.set_title("(d) Within-Question: Density Difference", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)


def plot_panel_steps_vs_acc(all_data, ax):
    """Model-level: avg steps vs accuracy with error bars and regression."""
    xs, ys, x_lo, x_hi, y_lo, y_hi, labels = [], [], [], [], [], [], []

    for model in MODEL_ORDER:
        if model not in all_data:
            continue
        records = all_data[model]

        all_steps = []
        all_correct = []
        for rec in records:
            for s in rec["samples"]:
                all_steps.append(s["n_steps"])
                all_correct.append(1.0 if s["correct"] else 0.0)

        s_mean, s_lo, s_hi = bootstrap_ci(all_steps)
        a_mean, a_lo, a_hi = bootstrap_ci(all_correct)

        xs.append(s_mean)
        ys.append(a_mean)
        x_lo.append(s_mean - s_lo)
        x_hi.append(s_hi - s_mean)
        y_lo.append(a_mean - a_lo)
        y_hi.append(a_hi - a_mean)
        labels.append(MODEL_SHORT.get(model, model.split("/")[-1]))

    xs, ys = np.array(xs), np.array(ys)

    ax.errorbar(xs, ys,
                xerr=[x_lo, x_hi], yerr=[y_lo, y_hi],
                fmt="s", markersize=10, capsize=4, capthick=1.5,
                color="#059669", ecolor="#6EE7B7", zorder=5)

    for i, label in enumerate(labels):
        ax.annotate(label, (xs[i], ys[i]),
                    textcoords="offset points", xytext=(8, 6),
                    fontsize=11, fontweight="bold")

    if len(xs) >= 3:
        slope, intercept, r_value, p_value, _ = stats.linregress(xs, ys)
        x_fit = np.linspace(xs.min() - 0.5, xs.max() + 0.5, 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, "--", color="#DC2626", alpha=0.7, linewidth=1.5)
        rho, p_spearman = stats.spearmanr(xs, ys)
        ax.text(0.05, 0.05,
                f"Spearman ρ = {rho:.2f} (p = {p_spearman:.3f})\n"
                f"Pearson r = {r_value:.2f} (p = {p_value:.3f})",
                transform=ax.transAxes, fontsize=9,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Avg Number of Reasoning Steps")
    ax.set_ylabel("Accuracy")
    ax.set_title("(e) Model-Level: Steps vs Accuracy", fontweight="bold")
    ax.grid(alpha=0.3)


def plot_summary_stats(all_data):
    """Print summary statistics table."""
    print("\n" + "=" * 90)
    print(f"{'Model':>30s} {'pass@1':>8s} {'Avg ρ':>8s} {'Avg Steps':>10s} "
          f"{'ρ (corr)':>10s} {'ρ (incorr)':>12s} {'Δρ':>8s}")
    print("=" * 90)

    for model in MODEL_ORDER:
        if model not in all_data:
            continue
        records = all_data[model]

        correct_d, incorrect_d, all_d, all_s, all_c = [], [], [], [], []
        for rec in records:
            for s in rec["samples"]:
                d = s["density_rho"]
                all_d.append(d)
                all_s.append(s["n_steps"])
                c = s["correct"]
                all_c.append(1.0 if c else 0.0)
                if c:
                    correct_d.append(d)
                else:
                    incorrect_d.append(d)

        short = MODEL_SHORT.get(model, model.split("/")[-1])
        pass1 = np.mean(all_c)
        avg_rho = np.mean(all_d)
        avg_steps = np.mean(all_s)
        rho_c = np.mean(correct_d) if correct_d else float("nan")
        rho_i = np.mean(incorrect_d) if incorrect_d else float("nan")
        delta = rho_c - rho_i

        print(f"{short:>30s} {pass1:>8.3f} {avg_rho:>8.1f} {avg_steps:>10.2f} "
              f"{rho_c:>10.1f} {rho_i:>12.1f} {delta:>+8.1f}")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="figure1_sampling_data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading sampling data...")
    all_data = load_sampling_data(data_dir)

    if not all_data:
        print("No data found. Exiting.")
        return

    plot_summary_stats(all_data)

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": "sans-serif",
    })

    # --- Full 6-panel figure ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    plot_panel_A(all_data, axes[0, 0])
    plot_panel_steps_vs_acc(all_data, axes[0, 1])
    plot_panel_B(all_data, axes[0, 2])
    plot_panel_C(all_data, axes[1, 0])
    plot_panel_D(all_data, axes[1, 1])

    # Panel F: steps of correct vs incorrect
    ax_f = axes[1, 2]
    for idx, model in enumerate(MODEL_ORDER):
        if model not in all_data:
            continue
        records = all_data[model]
        correct_steps = [s["n_steps"] for rec in records for s in rec["samples"] if s["correct"]]
        incorrect_steps = [s["n_steps"] for rec in records for s in rec["samples"] if not s["correct"]]
        short = MODEL_SHORT.get(model, model.split("/")[-1])
        x_pos = idx
        width = 0.35
        if correct_steps:
            bp1 = ax_f.boxplot([correct_steps], positions=[x_pos - width / 2],
                               widths=width * 0.8, patch_artist=True, showfliers=False,
                               boxprops=dict(facecolor="#BFDBFE", edgecolor="#2563EB"),
                               medianprops=dict(color="#1E40AF", linewidth=2))
        if incorrect_steps:
            bp2 = ax_f.boxplot([incorrect_steps], positions=[x_pos + width / 2],
                               widths=width * 0.8, patch_artist=True, showfliers=False,
                               boxprops=dict(facecolor="#FECACA", edgecolor="#DC2626"),
                               medianprops=dict(color="#991B1B", linewidth=2))
    ax_f.set_xticks(range(len([m for m in MODEL_ORDER if m in all_data])))
    ax_f.set_xticklabels([MODEL_SHORT.get(m, m.split("/")[-1])
                          for m in MODEL_ORDER if m in all_data], fontsize=9)
    ax_f.set_ylabel("Number of Steps")
    ax_f.set_title("(f) Within-Model: Steps Correct vs Incorrect", fontweight="bold")
    ax_f.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    full_path = OUT_DIR / "figure1_full_6panel.png"
    plt.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {full_path}")

    # --- Paper-ready 2-panel figure (replacement for original Figure 1) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_panel_A(all_data, ax1)
    ax1.set_title("(a) Reasoning Density vs Accuracy", fontweight="bold")
    plot_panel_steps_vs_acc(all_data, ax2)
    ax2.set_title("(b) Number of Steps vs Accuracy", fontweight="bold")
    plt.tight_layout()
    paper_path = OUT_DIR / "figure1_paper_2panel.png"
    plt.savefig(paper_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"Saved: {paper_path}")

    # --- Within-question analysis figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_panel_C(all_data, ax1)
    plot_panel_D(all_data, ax2)
    plt.tight_layout()
    within_path = OUT_DIR / "figure1_within_question.png"
    plt.savefig(within_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {within_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
