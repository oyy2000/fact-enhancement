#!/usr/bin/env python3
"""
E8 Calibration Size Sweep: Collect results and plot accuracy vs lambda.
Generates per-layer subplots with one line per calibration size.

Usage:
    python e8_plot_sweep.py \
        --mode GPT_REWRITE \
        [--mode LARGE_MODEL] \
        [--output_dir documents/e8_plots]
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

BASE = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement"
FONT_SIZE = 20
LEGEND_SIZE = 14

EVAL_DIRS = {
    "GPT_REWRITE": os.path.join(
        BASE, "calibration_ablation", "Qwen_Qwen2.5-3B-Instruct",
        "GPT_REWRITE", "eval"
    ),
    "LARGE_MODEL": os.path.join(
        BASE, "calibration_ablation", "Qwen_Qwen2.5-3B-Instruct",
        "LARGE_MODEL_Qwen_Qwen2.5-7B-Instruct", "eval"
    ),
}

FORMAL_STATUS_FILES = {
    "GPT_REWRITE": os.path.join(
        BASE, "calibration_ablation", "Qwen_Qwen2.5-3B-Instruct",
        "GPT_REWRITE", "formal_L6_limit1000_status.json"
    ),
    "LARGE_MODEL": os.path.join(
        BASE, "calibration_ablation", "Qwen_Qwen2.5-3B-Instruct",
        "LARGE_MODEL_Qwen_Qwen2.5-7B-Instruct", "formal_L6_limit1000_status.json"
    ),
}

SUMMARY_LABELS = {
    "GPT_REWRITE": "DenseSteer",
    "LARGE_MODEL": "InFamilySteer",
}

JOB_PATTERN = re.compile(
    r"Qwen2\.5-3B-Instruct_N(\d+)_L(\d+)_lam(n?[\dp]+)"
)


def parse_lambda(s):
    s = s.replace("p", ".").replace("n", "-")
    return float(s)


def collect_results(eval_dir):
    """Return list of dicts: {n, layer, lam, em}."""
    records = []
    if not os.path.isdir(eval_dir):
        print(f"[WARN] eval dir not found: {eval_dir}")
        return records

    for entry in os.listdir(eval_dir):
        m = JOB_PATTERN.match(entry)
        if not m:
            continue
        n = int(m.group(1))
        layer = int(m.group(2))
        lam = parse_lambda(m.group(3))

        result_dir = os.path.join(eval_dir, entry)
        em = None
        for root, dirs, files in os.walk(result_dir):
            for fn in files:
                if fn.startswith("results_") and fn.endswith(".json"):
                    fpath = os.path.join(root, fn)
                    try:
                        with open(fpath) as f:
                            data = json.load(f)
                        res = data.get("results", {})
                        for task_key, task_data in res.items():
                            if "exact_match,flexible-extract" in task_data:
                                em = task_data["exact_match,flexible-extract"]
                                break
                    except Exception:
                        pass
                    if em is not None:
                        break
            if em is not None:
                break

        if em is not None:
            records.append({"n": n, "layer": layer, "lam": lam, "em": em})

    return records


def read_exact_match(outdir):
    """Read exact-match accuracy from an lm-eval output directory."""
    if not outdir or not os.path.isdir(outdir):
        return None

    for root, dirs, files in os.walk(outdir):
        for fn in files:
            if not (fn.startswith("results_") and fn.endswith(".json")):
                continue
            fpath = os.path.join(root, fn)
            try:
                with open(fpath) as f:
                    data = json.load(f)
            except Exception:
                continue
            for task_data in data.get("results", {}).values():
                if "exact_match,flexible-extract" in task_data:
                    return task_data["exact_match,flexible-extract"]
    return None


def collect_formal_l6_results(mode):
    """Return final formal L6 results: {n, layer, lam, em}."""
    status_path = FORMAL_STATUS_FILES[mode]
    if not os.path.isfile(status_path):
        print(f"[WARN] formal status file not found: {status_path}")
        return []

    with open(status_path) as f:
        status = json.load(f)

    records = []
    for job in status.get("jobs", {}).values():
        if job.get("status") != "done":
            continue
        em = read_exact_match(job.get("outdir"))
        if em is None:
            em = job.get("pilot_em")
        if em is None:
            continue
        records.append({
            "n": int(job["calib_size"]),
            "layer": int(job["layer"]),
            "lam": float(job["lambda"]),
            "em": float(em),
        })

    records.sort(key=lambda r: r["n"])
    return records


def setup_plot_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "axes.titlesize": FONT_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def write_summary_csv(summary_by_mode, csv_path):
    with open(csv_path, "w") as f:
        f.write("method,N,layer,lambda,accuracy\n")
        for mode, records in summary_by_mode.items():
            label = SUMMARY_LABELS[mode]
            for r in records:
                f.write(f"{label},{r['n']},{r['layer']},{r['lam']},{r['em']}\n")


def plot_calibration_size_summary(summary_by_mode, output_dir, title=False):
    """Plot one summary figure: calibration size vs final GSM8K accuracy."""
    setup_plot_style()

    styles = {
        "GPT_REWRITE": {"color": "#1f77b4", "marker": "o"},
        "LARGE_MODEL": {"color": "#ff7f0e", "marker": "s"},
    }

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    all_accs = []
    xticks = [1, 5, 10, 25, 50, 100, 200]
    x_positions = {n: i for i, n in enumerate(xticks)}

    for mode in ["GPT_REWRITE", "LARGE_MODEL"]:
        records = summary_by_mode.get(mode, [])
        if not records:
            continue
        xs = [x_positions[r["n"]] for r in records]
        ys = [r["em"] * 100 for r in records]
        all_accs.extend(ys)
        style = styles[mode]
        ax.plot(
            xs, ys, "-",
            color=style["color"], marker=style["marker"],
            linewidth=2.6, markersize=8, label=SUMMARY_LABELS[mode],
        )

        peak_idx = int(np.argmax(ys))
        peak_x, peak_y = xs[peak_idx], ys[peak_idx]
        ax.annotate(
            f"Peak: {peak_y:.1f}%",
            xy=(peak_x, peak_y),
            xytext=(12, -10 if mode == "LARGE_MODEL" else 18),
            textcoords="offset points",
            fontsize=LEGEND_SIZE,
            color="#333333",
        )

    ax.set_xlim(-0.35, len(xticks) - 0.65)
    ax.set_xticks(range(len(xticks)))
    ax.set_xticklabels([str(x) for x in xticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Calibration set size N")
    ax.set_ylabel("GSM8K Accuracy (%)")
    if title:
        ax.set_title("Effect of Calibration Set Size on Steering Performance")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="lower left", frameon=True)

    if all_accs:
        ymin = np.floor((min(all_accs) - 0.2) * 2) / 2
        ymax = np.ceil((max(all_accs) + 0.2) * 2) / 2
        ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "e8_calibration_size_summary.png")
    pdf_path = os.path.join(output_dir, "e8_calibration_size_summary.pdf")
    csv_path = os.path.join(output_dir, "e8_calibration_size_summary.csv")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    write_summary_csv(summary_by_mode, csv_path)
    print(f"Saved: {out_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {csv_path}")
    return out_path


def plot_sweep(records, mode, output_dir):
    """Create plots: one figure per mode, subplots per layer, lines per N."""
    setup_plot_style()

    by_layer = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_layer[r["layer"]][r["n"]].append((r["lam"], r["em"]))

    layers = sorted(by_layer.keys())
    if not layers:
        print(f"[WARN] No results for {mode}")
        return

    n_layers = len(layers)
    fig_width = max(8, 6.4 * n_layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(fig_width, 5.8), squeeze=False)

    colors = {1: "#e41a1c", 5: "#377eb8", 10: "#4daf4a", 25: "#984ea3",
              50: "#ff7f00", 100: "#a65628", 200: "#f781bf"}

    for idx, layer in enumerate(layers):
        ax = axes[0][idx]
        n_data = by_layer[layer]
        sizes = sorted(n_data.keys())
        for n in sizes:
            pts = sorted(n_data[n], key=lambda x: x[0])
            lams = [p[0] for p in pts]
            ems = [p[1] for p in pts]
            color = colors.get(n, "#333333")
            ax.plot(lams, ems, "o-", label=f"N={n}", color=color, markersize=5, linewidth=2.0)

        ax.set_xlabel(rf"$\lambda$ (L{layer})")
        ax.set_ylabel("Exact Match")
        ax.legend(fontsize=LEGEND_SIZE, loc="best", frameon=True)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"e8_sweep_{mode}.png")
    pdf_path = os.path.join(output_dir, f"e8_sweep_{mode}.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Saved: {pdf_path}")
    return out_path


def print_best(records, mode):
    """Print top-5 configurations per calibration size."""
    by_n = defaultdict(list)
    for r in records:
        by_n[r["n"]].append(r)

    print(f"\n{'=' * 70}")
    print(f"  Best configs for {mode}")
    print(f"{'=' * 70}")

    for n in sorted(by_n.keys()):
        recs = sorted(by_n[n], key=lambda x: -x["em"])
        print(f"\n  N={n} (top 5):")
        print(f"    {'Layer':>5}  {'Lambda':>8}  {'EM':>6}")
        print(f"    {'-----':>5}  {'------':>8}  {'----':>6}")
        for r in recs[:5]:
            print(f"    {r['layer']:>5}  {r['lam']:>8.2f}  {r['em']:>6.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", nargs="+",
                        default=["GPT_REWRITE", "LARGE_MODEL"],
                        choices=["GPT_REWRITE", "LARGE_MODEL"])
    parser.add_argument("--output_dir", default=os.path.join(BASE, "documents", "e8_plots"))
    parser.add_argument("--single_plot", action="store_true",
                        help="Generate the single calibration-size summary plot from formal L6 results.")
    parser.add_argument("--single_only", action="store_true",
                        help="Only generate the single summary plot, skipping lambda-sweep subplots.")
    parser.add_argument("--summary_title", action="store_true",
                        help="Add the example-style title to the single summary plot.")
    args = parser.parse_args()

    if args.single_plot or args.single_only:
        summary_by_mode = {
            mode: collect_formal_l6_results(mode)
            for mode in args.mode
        }
        plot_calibration_size_summary(summary_by_mode, args.output_dir, title=args.summary_title)

    if args.single_only:
        return

    for mode in args.mode:
        eval_dir = EVAL_DIRS[mode]
        print(f"\n--- Collecting results for {mode} from {eval_dir} ---")
        records = collect_results(eval_dir)
        print(f"  Found {len(records)} result entries")

        if not records:
            continue

        sizes = sorted(set(r["n"] for r in records))
        layers = sorted(set(r["layer"] for r in records))
        print(f"  Sizes: {sizes}")
        print(f"  Layers: {layers}")
        for n in sizes:
            cnt = sum(1 for r in records if r["n"] == n)
            print(f"    N={n}: {cnt} results")

        plot_sweep(records, mode, args.output_dir)
        print_best(records, mode)


if __name__ == "__main__":
    main()
