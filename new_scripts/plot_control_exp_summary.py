#!/usr/bin/env python3
"""
Bar chart summary of control-experiment GSM8K accuracy
vs Baseline / DenseSteer / InFamilySteer.
Two panels: (A) Steering Methods  (B) E3 Rewriting Variants.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams.update({"font.size": 13, "font.family": "sans-serif"})

OUT = Path("/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/documents")
OUT.mkdir(parents=True, exist_ok=True)

# ── Data ────────────────────────────────────────────────────────────────────
# Panel A: main methods
methods_a = [
    "Baseline",
    "DenseSteer\n(L6, λ=4)",
    "InFamilySteer\n(L6, λ=0.45)",
    "E0: Random\nParaphrase (λ=5)",
    "E1: Step\nCompress (λ=−1.5)",
    "E2: Dense+\nSemWrong (λ=1)",
]
acc_a = [83.8, 85.9, 85.3, 84.6, 84.3, 84.3]

# Panel B: E3 rewriting
methods_b = [
    "Baseline\n(λ=0)",
    "E3: Rule-Based\nRewriting (λ=4)",
    "E3: GPT-5-mini\nRewriting (λ=1.5)",
]
acc_b = [83.8, 84.4, 84.7]

# ── Colors ──────────────────────────────────────────────────────────────────
# Baseline grey, DenseSteer blue, InFamily teal, controls orange-ish
colors_a = [
    "#9e9e9e",   # Baseline
    "#1976d2",   # DenseSteer
    "#00897b",   # InFamilySteer
    "#fb8c00",   # E0
    "#f4511e",   # E1
    "#8e24aa",   # E2
]
colors_b = [
    "#9e9e9e",   # Baseline
    "#5c6bc0",   # Rule-Based
    "#26a69a",   # GPT-5-mini
]

# ── Figure ──────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                gridspec_kw={"width_ratios": [6, 3]})

# --- Panel A ---
x_a = np.arange(len(methods_a))
bars_a = ax1.bar(x_a, acc_a, color=colors_a, width=0.6, edgecolor="white", linewidth=0.8)
ax1.set_xticks(x_a)
ax1.set_xticklabels(methods_a, fontsize=10)
ax1.set_ylabel("GSM8K Accuracy (%)", fontsize=13)
ax1.set_title("(A)  Steering Method Comparison", fontsize=14, fontweight="bold", pad=12)

# value labels
for bar, v in zip(bars_a, acc_a):
    delta = v - 83.8
    label = f"{v:.1f}%"
    if abs(delta) > 0.01:
        sign = "+" if delta > 0 else ""
        label += f"\n({sign}{delta:.1f})"
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
             label, ha="center", va="bottom", fontsize=9.5, fontweight="bold")

# horizontal reference lines
ax1.axhline(83.8, color="#9e9e9e", ls="--", lw=1, alpha=0.6, label="Baseline 83.8%")
ax1.axhline(85.9, color="#1976d2", ls=":", lw=1, alpha=0.5, label="DenseSteer 85.9%")
ax1.set_ylim(82.5, 87.2)
ax1.legend(fontsize=9, loc="upper right")

# --- Panel B ---
x_b = np.arange(len(methods_b))
bars_b = ax2.bar(x_b, acc_b, color=colors_b, width=0.55, edgecolor="white", linewidth=0.8)
ax2.set_xticks(x_b)
ax2.set_xticklabels(methods_b, fontsize=10)
ax2.set_title("(B)  E3: Token Rewriting Variants", fontsize=14, fontweight="bold", pad=12)

for bar, v in zip(bars_b, acc_b):
    delta = v - 83.8
    label = f"{v:.1f}%"
    if abs(delta) > 0.01:
        sign = "+" if delta > 0 else ""
        label += f"\n({sign}{delta:.1f})"
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
             label, ha="center", va="bottom", fontsize=9.5, fontweight="bold")

ax2.axhline(83.8, color="#9e9e9e", ls="--", lw=1, alpha=0.6)
ax2.axhline(85.9, color="#1976d2", ls=":", lw=1, alpha=0.5, label="DenseSteer 85.9%")
ax2.set_ylim(82.5, 87.2)
ax2.legend(fontsize=9, loc="upper right")

for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

fig.tight_layout(w_pad=3)
out_path = OUT / "control_exp_gsm8k_summary.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.close(fig)
