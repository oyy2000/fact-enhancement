#!/usr/bin/env python3
"""Standalone plot script for joint metrics from cache."""
import csv, json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

BASE = Path("/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement")
DOCS = BASE / "documents"
CACHE = DOCS / "joint_metrics_cache"
FIGURE7_DIR = BASE / "camera_ready" / "figure" / "figure7"

# Load all cache
results = []
for f in sorted(CACHE.glob("*.json")):
    blob = json.loads(f.read_text(encoding="utf-8"))
    if blob.get("n_scored", 0) > 0:
        results.append(blob)
print(f"Loaded {len(results)} results", flush=True)

# Save JSON
out_json = DOCS / "joint_metrics_grid_eval.json"
out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"wrote {out_json}", flush=True)

# Group by method
by_method = {}
for r in results:
    by_method.setdefault(r["method"], []).append(r)

METHOD_STYLES = {
    "FactSteer":      {"color": "#1a237e", "marker": "o"},
    "DenseSteer":     {"color": "#2e7d32", "marker": "s"},
    "InfamilySteer":  {"color": "#e65100", "marker": "^"},
    "E0":             {"color": "#6a1b9a", "marker": "D"},
    "E1":             {"color": "#ad1457", "marker": "v"},
    "E2":             {"color": "#00838f", "marker": "<"},
    "E3":             {"color": "#4e342e", "marker": ">"},
    "E3.2":           {"color": "#546e7a", "marker": "p"},
}

DAS_METHOD_STYLES = {
    "DenseSteer": {"color": "#062B63", "marker": "s"},
    "E0":         {"color": "#56B4E9", "marker": "D"},
    "E1":         {"color": "#0072B2", "marker": "v"},
    "E2":         {"color": "#3F51B5", "marker": "<"},
    "E3":         {"color": "#6A1B9A", "marker": ">"},
    "E3.2":       {"color": "#2477C3", "marker": "p"},
}

METHOD_LABELS = {
    "FactSteer":      "FactSteer",
    "DenseSteer":     "DenseSteer",
    "InfamilySteer":  "InfamilySteer",
    "E0":             "Random Paraphrase",
    "E1":             "Random Step Compression",
    "E2":             "Dense-but-Incorrect",
    "E3":             "Rule-Based Rewriting",
    "E3.2":           "GPT-5-mini Rewriting",
}

metrics = [
    ("mean_ppl", "Perplexity (PPL)", "PPL"),
    ("mean_rho", r"Reasoning Density ($\rho$)", "rho"),
    ("mean_c", r"Compatibility $c = \exp(-\mathrm{NLL})$", "compatibility"),
    ("mean_J", r"DenseCompat $J = \rho \cdot \exp(-\mathrm{NLL})$", "DenseCompat"),
    ("mean_DAS", r"DAS $= \log\rho - \mathrm{NLL}$", "DAS"),
    ("mean_JointZ", r"JointZ $= z_\rho - z_{\mathrm{NLL}}$", "JointZ"),
]

# ── Per-metric, all methods on one plot (best layer per method) ──
METHODS_ALL = ["FactSteer", "DenseSteer", "InfamilySteer", "E0", "E1", "E2", "E3", "E3.2"]
# DAS plot: exclude FactSteer and InfamilySteer; clip DenseSteer to [-6, 6]
DAS_EXCLUDE = {"FactSteer", "InfamilySteer"}
DAS_XLIM = {"DenseSteer": (-6, 6)}

def _plot_one_metric(ax, metric_key, fname, methods, by_method):
    plotted_rows = []
    for method in methods:
        recs = by_method.get(method, [])
        if not recs:
            continue
        style_map = DAS_METHOD_STYLES if fname == "DAS" else METHOD_STYLES
        style = style_map.get(method, {"color": "#333", "marker": "o"})
        by_layer = {}
        for r in recs:
            by_layer.setdefault(r["layer"], []).append(r)
        best_layer = max(by_layer.keys(), key=lambda L: len(by_layer[L]))
        recs_l = sorted(by_layer[best_layer], key=lambda r: r["lambda"])
        # Per-method x clipping for DAS
        if fname == "DAS" and method in DAS_XLIM:
            lo, hi = DAS_XLIM[method]
            recs_l = [r for r in recs_l if lo <= r["lambda"] <= hi]
        xs = [r["lambda"] for r in recs_l]
        ys = [r[metric_key] for r in recs_l]
        display = METHOD_LABELS.get(method, method)
        label = display if fname == "DAS" else f"{display} (L{best_layer})"
        ax.plot(xs, ys, marker=style["marker"], linewidth=1.5, markersize=4,
                color=style["color"], label=label)
        for r in recs_l:
            plotted_rows.append({
                "method": method,
                "label": display,
                "layer": best_layer,
                "lambda": r["lambda"],
                metric_key: r[metric_key],
            })
    return plotted_rows

for metric_key, ylabel, fname in metrics:
    if fname == "DAS":
        fig = plt.figure(figsize=(10, 8), dpi=300)
        ax = fig.add_axes([0.13, 0.12, 0.83, 0.66])
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
    methods = [m for m in METHODS_ALL if not (fname == "DAS" and m in DAS_EXCLUDE)]
    plotted_rows = _plot_one_metric(ax, metric_key, fname, methods, by_method)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if fname == "DAS":
        ax.set_xlim(-7.5, 7.5)
        ax.tick_params(axis="both", labelsize=20)
        ax.xaxis.label.set_size(20)
        ax.yaxis.label.set_size(20)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
                   ncol=2, frameon=False, fontsize=14)
    else:
        ax.legend(loc="best", fontsize=7, ncol=2)
        ax.set_title(f"{ylabel} vs Steering Strength")
        fig.tight_layout()
    outp = DOCS / f"joint_{fname}_vs_lambda.png"
    fig.savefig(outp, dpi=300 if fname == "DAS" else 200)
    if fname == "DAS":
        fig.savefig(DOCS / f"joint_{fname}_vs_lambda.pdf")
        csv_path = DOCS / f"joint_{fname}_vs_lambda.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["method", "label", "layer", "lambda", metric_key],
            )
            writer.writeheader()
            writer.writerows(plotted_rows)
        print(f"wrote {csv_path}", flush=True)
        FIGURE7_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE7_DIR / "figure7.png", dpi=300)
        fig.savefig(FIGURE7_DIR / "figure7.pdf")
        fig.savefig(FIGURE7_DIR / "joint_DAS_vs_lambda.png", dpi=300)
        fig.savefig(FIGURE7_DIR / "joint_DAS_vs_lambda.pdf")
        fig7_csv_path = FIGURE7_DIR / "figure7.csv"
        with fig7_csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["method", "label", "layer", "lambda", metric_key],
            )
            writer.writeheader()
            writer.writerows(plotted_rows)
        print(f"wrote {FIGURE7_DIR / 'figure7.png'}", flush=True)
        print(f"wrote {fig7_csv_path}", flush=True)
    plt.close()
    print(f"wrote {outp}", flush=True)

# ── DAS: separate InfamilySteer plot ──
LAYER_ALPHAS = {6: 0.5, 9: 0.55, 10: 0.6, 16: 0.65, 17: 0.8, 18: 0.7, 27: 0.85, 31: 0.9, 35: 1.0}

if "InfamilySteer" in by_method:
    metric_key, ylabel, fname = "mean_DAS", r"DAS $= \log\rho - \mathrm{NLL}$", "DAS"
    recs = by_method["InfamilySteer"]
    by_layer_inf = {}
    for r in recs:
        by_layer_inf.setdefault(r["layer"], []).append(r)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for L in sorted(by_layer_inf.keys()):
        recs_l = sorted(by_layer_inf[L], key=lambda r: r["lambda"])
        xs = [r["lambda"] for r in recs_l]
        ys = [r[metric_key] for r in recs_l]
        alpha = LAYER_ALPHAS.get(L, 0.7)
        ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3,
                alpha=alpha, label=f"L{L}")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(r"InfamilySteer: DAS vs $\lambda$")
    fig.tight_layout()
    outp = DOCS / "joint_DAS_InfamilySteer_standalone.png"
    fig.savefig(outp, dpi=200)
    plt.close()
    print(f"wrote {outp}", flush=True)

# ── Per-method multi-layer plots for DenseCompat and DAS ──
key_metrics = [
    ("mean_J", r"DenseCompat $J$", "DenseCompat"),
    ("mean_DAS", r"DAS", "DAS"),
]

for method, recs in sorted(by_method.items()):
    by_layer = {}
    for r in recs:
        by_layer.setdefault(r["layer"], []).append(r)
    if len(by_layer) <= 1:
        continue
    for metric_key, ylabel, fname in key_metrics:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for L in sorted(by_layer.keys()):
            recs_l = sorted(by_layer[L], key=lambda r: r["lambda"])
            xs = [r["lambda"] for r in recs_l]
            ys = [r[metric_key] for r in recs_l]
            alpha = LAYER_ALPHAS.get(L, 0.7)
            ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3,
                    alpha=alpha, label=f"L{L}")
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        ax.set_title(f"{method}: {ylabel} vs " + r"$\lambda$")
        fig.tight_layout()
        outp = DOCS / f"joint_{fname}_{method}_multilayer.png"
        fig.savefig(outp, dpi=200)
        plt.close()
        print(f"wrote {outp}", flush=True)

print("ALL_PLOTS_DONE", flush=True)
