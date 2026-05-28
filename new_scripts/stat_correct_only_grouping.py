#!/usr/bin/env python3
"""
Statistical analysis: are the proposed model groups truly similar within
and different between?

Methods:
  1. Per-dataset mean ρ table  →  visual sanity check
  2. Cohen's d (effect size) between groups, per dataset
  3. Mann-Whitney U test between groups, per dataset
  4. Permutation test for group-mean difference
  5. Hierarchical clustering dendrogram on the (model × dataset) ρ matrix
  6. Silhouette score for the proposed 2-group partition
"""

import json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

def short_name(d):
    n = d.replace("Qwen_Qwen2.5-", "Qwen-").replace("-Instruct", "")
    n = n.replace("meta-llama_Llama-3.2-", "Llama-").replace("meta-llama_Llama-3.1-", "Llama-")
    return n

def ds_label(ds):
    return {"gsm8k":"GSM8K","math500":"MATH-500","aime":"AIME","amc":"AMC","olympiad":"Olympiad"}.get(ds, ds)

# ── Load data ─────────────────────────────────────────────────────────────────
raw = defaultdict(lambda: defaultdict(lambda: {"rho": [], "steps": []}))

def load(path, dsl, ms):
    if ms in EXCLUDE_MODELS:
        return
    with open(path) as f:
        for line in f:
            doc = json.loads(line)
            for s in doc["samples"]:
                if s["correct"]:
                    raw[ms][dsl]["rho"].append(s["density_rho"])
                    raw[ms][dsl]["steps"].append(s["n_steps"])

for d in sorted(GSM_DIR.iterdir()):
    p = d / "gsm8k_samples.jsonl"
    if p.exists(): load(p, "GSM8K", short_name(d.name))

for ds in DATASETS_MULTI:
    dd = MULTI_DIR / ds
    if not dd.exists(): continue
    for d in sorted(dd.iterdir()):
        p = d / "samples.jsonl"
        if p.exists(): load(p, ds_label(ds), short_name(d.name))

models_present = [m for m in MODEL_ORDER if m in raw]

# ── Group definitions ─────────────────────────────────────────────────────────
FAMILIES = {
    "Qwen": {
        "models": [m for m in models_present if m.startswith("Qwen") and m != "Qwen-0.5B"],
        "groups": {
            "≤7B":  [m for m in models_present if m.startswith("Qwen") and m != "Qwen-0.5B" and
                      float(m.split("-")[1].replace("B","")) <= 7],
            "≥14B": [m for m in models_present if m.startswith("Qwen") and
                      float(m.split("-")[1].replace("B","")) >= 14],
        },
    },
    "Llama": {
        "models": [m for m in models_present if m.startswith("Llama")],
        "groups": {
            "≤3B": [m for m in models_present if m.startswith("Llama") and
                     float(m.split("-")[1].replace("B","")) <= 3],
            "8B":  [m for m in models_present if m.startswith("Llama") and
                     float(m.split("-")[1].replace("B","")) >= 8],
        },
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# 1. Build per-model mean-ρ matrix (for clustering & silhouette)
# ══════════════════════════════════════════════════════════════════════════════
for family_name, fam in FAMILIES.items():
    models = fam["models"]
    groups = fam["groups"]
    group_names = list(groups.keys())
    if len(models) < 2:
        continue

    print(f"\n{'='*72}")
    print(f"  Family: {family_name}")
    print(f"{'='*72}")

    # ── Mean ρ matrix ─────────────────────────────────────────────────────
    ds_used = [ds for ds in DATASETS_ALL
               if any(len(raw[m][ds]["rho"]) > 0 for m in models)]
    mat = np.zeros((len(models), len(ds_used)))
    for i, m in enumerate(models):
        for j, ds in enumerate(ds_used):
            arr = raw[m][ds]["rho"]
            mat[i, j] = np.mean(arr) if len(arr) > 0 else np.nan

    df_mean = pd.DataFrame(mat, index=models, columns=ds_used)
    print("\n── Per-model mean ρ (correct only) ──")
    print(df_mean.round(1).to_string())

    # ── 2. Cohen's d & Mann-Whitney U per dataset ────────────────────────
    print(f"\n── Cohen's d & Mann-Whitney U  ({' vs '.join(group_names)}) ──")
    print(f"{'Dataset':>12s} | {'Cohen d':>9s} | {'U-stat':>10s} | {'p-value':>10s} | {'Interpretation':>20s}")
    print("-" * 72)

    for ds in ds_used:
        # pool all correct-sample ρ values per group
        g1_vals = np.concatenate([raw[m][ds]["rho"] for m in groups[group_names[0]]
                                  if len(raw[m][ds]["rho"]) > 0])
        g2_vals = np.concatenate([raw[m][ds]["rho"] for m in groups[group_names[1]]
                                  if len(raw[m][ds]["rho"]) > 0])
        if len(g1_vals) == 0 or len(g2_vals) == 0:
            print(f"{ds:>12s} | {'N/A':>9s} | {'N/A':>10s} | {'N/A':>10s} |")
            continue

        # Cohen's d
        pooled_std = np.sqrt(((len(g1_vals)-1)*np.var(g1_vals, ddof=1) +
                               (len(g2_vals)-1)*np.var(g2_vals, ddof=1)) /
                              (len(g1_vals) + len(g2_vals) - 2))
        d = (np.mean(g1_vals) - np.mean(g2_vals)) / pooled_std if pooled_std > 0 else 0

        # Mann-Whitney U
        u_stat, u_p = sp_stats.mannwhitneyu(g1_vals, g2_vals, alternative="two-sided")

        interp = ("large" if abs(d) >= 0.8 else
                  "medium" if abs(d) >= 0.5 else
                  "small" if abs(d) >= 0.2 else "negligible")
        sig = "***" if u_p < 0.001 else "**" if u_p < 0.01 else "*" if u_p < 0.05 else "ns"

        print(f"{ds:>12s} | {d:>+9.3f} | {u_stat:>10.0f} | {u_p:>10.2e} | d={interp}, {sig}")

    # ── 3. Permutation test on group-mean difference (across datasets) ───
    print(f"\n── Permutation test (10000 iters): mean ρ difference across datasets ──")
    # For each model, compute its mean ρ across datasets (the "profile mean")
    profile_means = {}
    for m in models:
        vals = [np.mean(raw[m][ds]["rho"]) for ds in ds_used if len(raw[m][ds]["rho"]) > 0]
        profile_means[m] = np.mean(vals) if vals else np.nan

    g1_profile = [profile_means[m] for m in groups[group_names[0]] if not np.isnan(profile_means[m])]
    g2_profile = [profile_means[m] for m in groups[group_names[1]] if not np.isnan(profile_means[m])]

    obs_diff = abs(np.mean(g1_profile) - np.mean(g2_profile))
    all_profiles = np.array(g1_profile + g2_profile)
    n1 = len(g1_profile)
    n_perm = 10000
    count = 0
    rng = np.random.default_rng(42)
    for _ in range(n_perm):
        perm = rng.permutation(all_profiles)
        perm_diff = abs(np.mean(perm[:n1]) - np.mean(perm[n1:]))
        if perm_diff >= obs_diff:
            count += 1
    perm_p = count / n_perm
    print(f"  Group means: {group_names[0]}={np.mean(g1_profile):.2f}, "
          f"{group_names[1]}={np.mean(g2_profile):.2f}")
    print(f"  Observed |Δ| = {obs_diff:.2f},  permutation p = {perm_p:.4f}")

    # ── 4. Hierarchical clustering dendrogram ────────────────────────────
    # Use the mean-ρ profile across datasets as feature vector
    valid_mask = ~np.isnan(mat).any(axis=1)
    mat_clean = mat[valid_mask]
    models_clean = [models[i] for i in range(len(models)) if valid_mask[i]]

    if len(models_clean) >= 2:
        Z = linkage(mat_clean, method="ward", metric="euclidean")

        fig, ax = plt.subplots(figsize=(8, 4.5))
        # Color by group
        group_labels = []
        for m in models_clean:
            assigned = "?"
            for gn, gm in groups.items():
                if m in gm:
                    assigned = gn
                    break
            group_labels.append(assigned)

        dn = dendrogram(Z, labels=models_clean, ax=ax, leaf_rotation=30,
                        leaf_font_size=10, above_threshold_color="0.5")

        # Annotate leaves with group color
        colors_map = {group_names[0]: "#5B9BD5" if family_name == "Qwen" else "#ED7D31",
                      group_names[1]: "#2E5A88" if family_name == "Qwen" else "#A04000"}
        xlbls = ax.get_xticklabels()
        for lbl in xlbls:
            m = lbl.get_text()
            for gn, gm in groups.items():
                if m in gm:
                    lbl.set_color(colors_map.get(gn, "black"))
                    lbl.set_fontweight("bold")
                    break

        ax.set_ylabel("Ward distance")
        ax.set_title(f"{family_name} — Hierarchical Clustering on Mean ρ Profile (Correct Only)",
                     fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fname = OUT_DIR / f"dendrogram_{family_name}_rho.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"  Saved dendrogram → {fname}")
        plt.close(fig)

    # ── 5. Silhouette score ──────────────────────────────────────────────
    if len(models_clean) >= 3:
        # Assign numeric labels: group0=0, group1=1
        labels_num = []
        for m in models_clean:
            for gi, (gn, gm) in enumerate(groups.items()):
                if m in gm:
                    labels_num.append(gi)
                    break
        labels_num = np.array(labels_num)

        if len(np.unique(labels_num)) >= 2:
            sil = silhouette_score(mat_clean, labels_num, metric="euclidean")
            print(f"\n  Silhouette score (proposed grouping): {sil:.3f}")
            print(f"    Interpretation: {'strong' if sil > 0.5 else 'reasonable' if sil > 0.25 else 'weak'} structure")

            # Per-model silhouette
            from sklearn.metrics import silhouette_samples
            sil_samples = silhouette_samples(mat_clean, labels_num, metric="euclidean")
            print(f"  Per-model silhouette:")
            for m, s in zip(models_clean, sil_samples):
                print(f"    {m:>12s}: {s:+.3f}")

    # ── 6. Spearman correlation between model-size and mean ρ ────────────
    print(f"\n── Spearman rank correlation: model size vs mean ρ ──")
    sizes = []
    mean_rhos = []
    for m in models:
        sz = float(m.split("-")[1].replace("B",""))
        mr = profile_means.get(m, np.nan)
        if not np.isnan(mr):
            sizes.append(sz)
            mean_rhos.append(mr)
    if len(sizes) >= 3:
        rho_corr, rho_p = sp_stats.spearmanr(sizes, mean_rhos)
        print(f"  Spearman ρ = {rho_corr:.3f}, p = {rho_p:.4f}")
    else:
        print("  Not enough data points")

print("\n" + "="*72)
print("All done.")
