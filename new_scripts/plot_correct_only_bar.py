#!/usr/bin/env python3
"""
Bar charts: Avg ρ (tok/step) and Avg Steps for ALL samples,
x-axis = model (per family), one panel per dataset + Average.
Grouped coloring + bracket annotations.
Excludes Qwen-0.5B, Qwen-72B, Llama-70B.
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
matplotlib.rcParams.update({"font.size": 12})

BASE = Path("/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement")
OUT_DIR = BASE / "documents" / "e4_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MULTI_DIR = BASE / "new_exps" / "figure1_multi_dataset"
GSM_DIR   = BASE / "new_exps" / "figure1_sampling_data"

DATASETS_MULTI = ["math500", "aime", "amc", "olympiad"]
DATASETS_ALL   = ["GSM8K", "MATH-500", "AIME", "AMC", "Olympiad"]

MODEL_ORDER = [
    "Qwen-0.5B", "Llama-1B", "Qwen-1.5B",
    "Qwen-3B", "Llama-3B",
    "Qwen-7B", "Llama-8B",
    "Qwen-14B", "Qwen-32B",
]
EXCLUDE_MODELS = {"Qwen-72B", "Llama-70B"}


def short_name(model_dir: str) -> str:
    name = model_dir.replace("Qwen_Qwen2.5-", "Qwen-").replace("-Instruct", "")
    name = name.replace("meta-llama_Llama-3.2-", "Llama-").replace("meta-llama_Llama-3.1-", "Llama-")
    return name


def dataset_label(ds: str) -> str:
    return {"gsm8k": "GSM8K", "math500": "MATH-500", "aime": "AIME",
            "amc": "AMC", "olympiad": "Olympiad"}.get(ds, ds)


# ── Load ALL samples (not just correct) ───────────────────────────────────────
result_all = defaultdict(lambda: defaultdict(lambda: {"rho": [], "steps": []}))
result_correct = defaultdict(lambda: defaultdict(lambda: {"rho": [], "steps": []}))


def load_samples(path, ds_label, model_short):
    if model_short in EXCLUDE_MODELS:
        return
    with open(path) as f:
        for line in f:
            doc = json.loads(line)
            for s in doc["samples"]:
                result_all[model_short][ds_label]["rho"].append(s["density_rho"])
                result_all[model_short][ds_label]["steps"].append(s["n_steps"])
                if s["correct"]:
                    result_correct[model_short][ds_label]["rho"].append(s["density_rho"])
                    result_correct[model_short][ds_label]["steps"].append(s["n_steps"])


for model_dir in sorted(GSM_DIR.iterdir()):
    gsm_file = model_dir / "gsm8k_samples.jsonl"
    if gsm_file.exists():
        load_samples(gsm_file, "GSM8K", short_name(model_dir.name))

for ds in DATASETS_MULTI:
    ds_dir = MULTI_DIR / ds
    if not ds_dir.exists():
        continue
    for model_dir in sorted(ds_dir.iterdir()):
        sf = model_dir / "samples.jsonl"
        if sf.exists():
            load_samples(sf, dataset_label(ds), short_name(model_dir.name))

models_present = [m for m in MODEL_ORDER if m in result_all]
print("Models found:", models_present)
for m in models_present:
    for ds in DATASETS_ALL:
        n = len(result_all[m][ds]["rho"])
        nc = len(result_correct[m][ds]["rho"])
        if n > 0:
            print(f"  {m:>12s} | {ds:>10s} | n={n:5d} (correct={nc:5d}) | "
                  f"avg_rho={np.mean(result_all[m][ds]['rho']):6.1f} | "
                  f"avg_steps={np.mean(result_all[m][ds]['steps']):5.2f}")

# ── Group definitions (Qwen: drop 0.5B) ──────────────────────────────────────
QWEN_GROUPS = [
    {"models": ["Qwen-1.5B", "Qwen-3B", "Qwen-7B"],
     "label": "1.5B–7B", "color": "#5B9BD5"},
    {"models": ["Qwen-14B", "Qwen-32B"],
     "label": "14B–32B", "color": "#2E5A88"},
]
LLAMA_GROUPS = [
    {"models": ["Llama-1B", "Llama-3B"],
     "label": "1B–3B", "color": "#ED7D31"},
    {"models": ["Llama-8B"],
     "label": "8B", "color": "#A04000"},
]

GAP = 0.7


def build_x_positions(groups, models_set):
    x_pos, colors, hatches, labels, group_spans = [], [], [], [], []
    pos = 0
    for gi, grp in enumerate(groups):
        grp_models = [m for m in grp["models"] if m in models_set]
        if not grp_models:
            continue
        if gi > 0 and len(x_pos) > 0:
            pos += GAP
        x_start = pos
        for m in grp_models:
            x_pos.append(pos)
            colors.append(grp["color"])
            hatches.append("" if gi == 0 else "//")
            labels.append(m.replace("Qwen-", "").replace("Llama-", ""))
            pos += 1
        x_end = pos - 1
        group_spans.append((x_start, x_end, grp["label"], grp["color"]))
    return np.array(x_pos), colors, hatches, labels, group_spans


def draw_bracket(ax, x1, x2, label, color, y_frac=-0.18):
    trans = ax.get_xaxis_transform()
    y0, y_text = y_frac, y_frac - 0.09
    ax.plot([x1 - 0.2, x2 + 0.2], [y0, y0], transform=trans,
            color=color, lw=2, clip_on=False)
    ax.plot([x1 - 0.2, x1 - 0.2], [y0, y0 + 0.025], transform=trans,
            color=color, lw=2, clip_on=False)
    ax.plot([x2 + 0.2, x2 + 0.2], [y0, y0 + 0.025], transform=trans,
            color=color, lw=2, clip_on=False)
    ax.text((x1 + x2) / 2, y_text, label, transform=trans,
            ha="center", va="top", fontsize=9, fontweight="bold", color=color)


def make_bar_grid(groups, family_tag, metric_key, ylabel, data, suffix, title_tag):
    """Bar chart grid: one panel per dataset + Average."""
    all_models = []
    for grp in groups:
        all_models.extend(grp["models"])
    models = [m for m in all_models if m in data]

    datasets_to_plot = [ds for ds in DATASETS_ALL
                        if any(len(data[m][ds][metric_key]) > 0 for m in models)]
    panels = datasets_to_plot + ["Average"]
    n_panels = len(panels)
    if len(models) == 0 or n_panels == 0:
        return

    x_pos, bar_colors, bar_hatches, short_labels, group_spans = \
        build_x_positions(groups, set(models))

    fig, axes = plt.subplots(1, n_panels, figsize=(3.4 * n_panels, 5.0), sharey=False)
    if n_panels == 1:
        axes = [axes]

    bar_width = 0.6

    # Pre-compute per-model average across datasets
    model_avg = {}
    for m in models:
        ds_means = [np.mean(data[m][ds][metric_key])
                    for ds in datasets_to_plot if len(data[m][ds][metric_key]) > 0]
        model_avg[m] = np.mean(ds_means) if ds_means else np.nan

    for ax, panel in zip(axes, panels):
        vals, errs = [], []
        for m in models:
            if panel == "Average":
                v = model_avg[m]
                vals.append(v if not np.isnan(v) else 0)
                ds_means = [np.mean(data[m][ds][metric_key])
                            for ds in datasets_to_plot if len(data[m][ds][metric_key]) > 0]
                errs.append(np.std(ds_means) / np.sqrt(len(ds_means)) if len(ds_means) > 1 else 0)
            else:
                arr = data[m][panel][metric_key]
                if len(arr) > 0:
                    vals.append(np.mean(arr))
                    errs.append(np.std(arr) / np.sqrt(len(arr)))
                else:
                    vals.append(0)
                    errs.append(0)

        # Draw bars
        for i in range(len(models)):
            ax.bar(x_pos[i], vals[i], bar_width, yerr=errs[i],
                   color=bar_colors[i], capsize=2, alpha=0.88,
                   edgecolor="white", linewidth=0.8,
                   hatch=bar_hatches[i])

        # Value labels on top
        for i, (v, e) in enumerate(zip(vals, errs)):
            if v > 0:
                ax.text(x_pos[i], v + e + 0.5, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(panel, fontsize=11, fontweight="bold",
                     color="#333" if panel != "Average" else "#B22222")
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(x_pos[0] - 0.5, x_pos[-1] + 0.7)

        for x1, x2, glabel, gcol in group_spans:
            draw_bracket(ax, x1, x2, glabel, gcol)

    axes[0].set_ylabel(ylabel, fontsize=11)

    # Legend
    legend_handles = []
    seen = set()
    for gi, grp in enumerate(groups):
        if grp["label"] not in seen:
            seen.add(grp["label"])
            legend_handles.append(
                mpatches.Patch(facecolor=grp["color"],
                               hatch="" if gi == 0 else "//",
                               edgecolor="white", alpha=0.88,
                               label=grp["label"]))
    axes[-1].legend(handles=legend_handles, loc="upper right", fontsize=9,
                    framealpha=0.9)

    fig.suptitle(f"{family_tag} — {ylabel} ({title_tag})",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])

    fname = f"grid_{family_tag}_{metric_key}{suffix}.png"
    out = OUT_DIR / fname
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


# ── Generate both versions ────────────────────────────────────────────────────
for data, suffix, tag in [(result_all, "", "All Samples"),
                           (result_correct, "_correct", "Correct Only")]:
    make_bar_grid(QWEN_GROUPS, "Qwen", "rho", "Avg ρ (tok/step)", data, suffix, tag)
    make_bar_grid(QWEN_GROUPS, "Qwen", "steps", "Avg Steps", data, suffix, tag)
    make_bar_grid(LLAMA_GROUPS, "Llama", "rho", "Avg ρ (tok/step)", data, suffix, tag)
    make_bar_grid(LLAMA_GROUPS, "Llama", "steps", "Avg Steps", data, suffix, tag)

print("Done.")
