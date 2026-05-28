#!/usr/bin/env python3
"""
Side-by-side visual diff of flipped samples (wrong->correct).
Highlights the critical reasoning change in each pair.
Uses FontProperties to force Noto Sans CJK SC for all text.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

# ── force CJK font via FontProperties ──────────────────────────────────────
_cjk_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FP      = fm.FontProperties(fname=_cjk_path, size=10)
FP_BOLD = fm.FontProperties(fname=_cjk_path, size=11.5)
FP_BOLD._weight = "bold"
FP_MONO = fm.FontProperties(fname=_cjk_path, size=10)
FP_SM   = fm.FontProperties(fname=_cjk_path, size=9)
FP_SM_I = fm.FontProperties(fname=_cjk_path, size=9, style="italic")
FP_TITLE = fm.FontProperties(fname=_cjk_path, size=14)
FP_TITLE._weight = "bold"

BASE = Path("/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement")
OUT  = BASE / "documents"

with open(OUT / "flipped_rho_up_steps_down.json") as f:
    samples = json.load(f)

# ── per-sample annotations ──────────────────────────────────────────────────
annotations = {
    531: {
        "error": '"3 times fewer" misread as 3x21=63 (multiply)',
        "fix":   '"3 times fewer" correctly as 21/3=7 (divide)',
        "bl_key": '3 x 21 = 63  ->  80-21+63 = 122  X',
        "st_key": '21 / 3 = 7   ->  59+7 = 66       OK',
    },
    570: {
        "error": '$10/day buy $5 figures -> count=4, $50/$2=25, 4+25=29',
        "fix":   '$10x4days=$40 -> 40/$5=8, $30/$2=15, 8+15=23',
        "bl_key": '4x$5=$20 spent -> $50 left -> 50/2=25 -> 4+25=29  X',
        "st_key": '4x$10=$40 spent -> 40/5=8 -> $30 left -> 30/2=15 -> 8+15=23  OK',
    },
    924: {
        "error": '4h down + 4h up = 8h/trip -> only 1 trip -> 24/1=24',
        "fix":   '4h per trip (incl. up+down) -> 8h/4h=2 trips -> 24/2=12',
        "bl_key": '4+4=8h per trip -> 1 trip -> 24/1 = 24  X',
        "st_key": '4h per trip -> 8/4 = 2 trips -> 24/2 = 12  OK',
    },
    943: {
        "error": 'Net loss = original $75 - sell $40 = 35 (used original, not paid)',
        "fix":   'Net loss = paid $90 - sell $40 = 50 (correctly used paid price)',
        "bl_key": 'loss = 75 - 40 = 35  X',
        "st_key": 'loss = 90 - 40 = 50  OK',
    },
    972: {
        "error": '1/4 x 12 (total) = 3 kept -> sell 5 -> 5x20=100',
        "fix":   '1/4 x 8 (remaining) = 2 kept -> sell 6 -> 6x20=120',
        "bl_key": '1/4 x 12 = 3 kept -> sell 8-3=5 -> 5x20 = 100  X',
        "st_key": '1/4 x 8 = 2 kept -> sell 8-2=6 -> 6x20 = 120  OK',
    },
}

# ── figure ──────────────────────────────────────────────────────────────────
N = len(samples)
ROW_H = 3.6
fig, axes = plt.subplots(N, 1, figsize=(17, N * ROW_H))
if N == 1:
    axes = [axes]

def txt(ax, x, y, s, fp=FP, color="black", **kw):
    """Helper: place text with explicit FontProperties."""
    ax.text(x, y, s, fontproperties=fp, color=color,
            transform=ax.transAxes, **kw)

for ax, s in zip(axes, samples):
    did = s["doc_id"]
    ann = annotations[did]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── header ──
    q_short = s["question"][:105] + ("..." if len(s["question"]) > 105 else "")
    header = (f"doc_id={did}    Answer={s['target']}    "
              f"rho: {s['bl_rho']} -> {s['st_rho']}  (D+{s['delta_rho']})    "
              f"steps: {s['bl_steps']} -> {s['st_steps']}")
    txt(ax, 0.0, 0.97, header, fp=FP_BOLD, va="top")
    txt(ax, 0.0, 0.82, f"Q: {q_short}", fp=FP_SM_I, color="#555", va="top")

    # ── left: baseline (red box) ──
    bl_box = mpatches.FancyBboxPatch(
        (0.01, 0.04), 0.47, 0.62, boxstyle="round,pad=0.015",
        facecolor="#fff0f0", edgecolor="#e53935", linewidth=1.8,
        transform=ax.transAxes)
    ax.add_patch(bl_box)
    txt(ax, 0.03, 0.64, "Baseline  X  WRONG", fp=FP_BOLD, color="#c62828", va="top")
    txt(ax, 0.03, 0.53, ann["bl_key"], fp=FP_MONO, color="#b71c1c", va="top")
    txt(ax, 0.03, 0.36, f"Error: {ann['error']}", fp=FP_SM, color="#d32f2f", va="top")

    # ── right: steered (green box) ──
    st_box = mpatches.FancyBboxPatch(
        (0.52, 0.04), 0.47, 0.62, boxstyle="round,pad=0.015",
        facecolor="#f0fff0", edgecolor="#2e7d32", linewidth=1.8,
        transform=ax.transAxes)
    ax.add_patch(st_box)
    txt(ax, 0.54, 0.64, "Steered  OK  CORRECT", fp=FP_BOLD, color="#1b5e20", va="top")
    txt(ax, 0.54, 0.53, ann["st_key"], fp=FP_MONO, color="#2e7d32", va="top")
    txt(ax, 0.54, 0.36, f"Fix: {ann['fix']}", fp=FP_SM, color="#388e3c", va="top")

    # ── arrow ──
    ax.annotate("", xy=(0.515, 0.40), xytext=(0.495, 0.40),
                arrowprops=dict(arrowstyle="-|>", color="#1976d2", lw=2.8,
                                mutation_scale=18),
                xycoords="axes fraction")

    # ── separator ──
    ax.plot([0.02, 0.98], [-0.01, -0.01], color="#ccc", lw=1,
            transform=ax.transAxes, clip_on=False)

fig.text(0.5, 1.01,
         "Flipped Samples:  Wrong -> Correct  +  rho up  +  Steps down\n"
         "DenseSteer L6 lam=4  vs  Baseline  |  Qwen2.5-3B-Instruct  |  GSM8K",
         fontproperties=FP_TITLE, ha="center", va="bottom")
fig.tight_layout(h_pad=1.2)
out_path = OUT / "flipped_samples_diff.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved -> {out_path}")
plt.close(fig)
