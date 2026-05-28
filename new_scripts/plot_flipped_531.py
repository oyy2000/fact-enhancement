#!/usr/bin/env python3
"""
Single-sample diff card for doc_id=531, showing only the Second Stop
where the critical reasoning error occurs.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

_font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FP    = fm.FontProperties(fname=_font, size=11)
FP_B  = fm.FontProperties(fname=_font, size=12.5, weight="bold")
FP_S  = fm.FontProperties(fname=_font, size=10)
FP_T  = fm.FontProperties(fname=_font, size=14, weight="bold")
FP_M  = fm.FontProperties(fname=_font, size=10.5)  # mono-ish

OUT = Path("/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/documents")

# ── text blocks ─────────────────────────────────────────────────────────────
baseline_lines = [
    ("Second Stop:", True),
    ("", False),
    ("- 21 people get off the bus.", False),
    ("- The number of people getting on the bus is", False),
    ('  3 times fewer than those who got off, which', False),
    ('  means they get on 3 times more than those', False),
    ('  who got off, i.e.,  3 × 21 = 63.', False),
    ("", False),
    ("After the second stop:", False),
    ("- People on the bus: 80 − 21 + 63 = 122.", False),
    ("", False),
    ("Final Answer:  122   [WRONG]", True),
]

steered_lines = [
    ("Second Stop:", True),
    ("", False),
    ("- 21 passengers get off the bus.", False),
    ("- Number of passengers left on the bus:", False),
    ("      80 − 21 = 59", False),
    ("- 3 times fewer passengers get on the bus", False),
    ("  than those who got off (21):", False),
    ("      21 / 3 = 7", False),
    ("- Number of passengers after the second stop:", False),
    ("      59 + 7 = 66", False),
    ("", False),
    ("Final Answer:  66  ✓", True),
]

# highlight indices (0-based) — the critical wrong/right lines
bl_highlight = {3, 4, 5, 6, 9}   # "3 times fewer … 63" + result
st_highlight = {5, 6, 7, 9}      # "3 times fewer … 7"  + result

# ── figure ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 6.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

# title
ax.text(0.5, 0.97,
        "doc_id = 531  |  Baseline → Steered  |  ρ: 29.5 → 59.4  |  steps: 8 → 5",
        fontproperties=FP_T, ha="center", va="top", transform=ax.transAxes)
ax.text(0.5, 0.91,
        "Q: 48 people are riding a bus. On the first stop, 8 get off and 5× as many get on …\n"
        "On the second stop, 21 get off and 3 times fewer get on. How many passengers remain?   (Answer = 66)",
        fontproperties=FP_S, ha="center", va="top", color="#444",
        transform=ax.transAxes, linespacing=1.4)

# ── left panel: Baseline ────────────────────────────────────────────────────
bl_box = mpatches.FancyBboxPatch(
    (0.02, 0.06), 0.46, 0.72, boxstyle="round,pad=0.02",
    facecolor="#fff5f5", edgecolor="#e53935", linewidth=2,
    transform=ax.transAxes)
ax.add_patch(bl_box)
ax.text(0.04, 0.76, "Baseline   [WRONG]",
        fontproperties=FP_B, color="#c62828", va="top", transform=ax.transAxes)

y = 0.70
for i, (line, bold) in enumerate(baseline_lines):
    fp = FP_B if bold else FP_M
    c = "#b71c1c" if i in bl_highlight else "#333"
    if i in bl_highlight:
        # subtle highlight background
        ax.axhspan(0, 1)  # dummy; we'll use a rectangle
        rect = mpatches.FancyBboxPatch(
            (0.03, y - 0.005), 0.44, 0.038, boxstyle="round,pad=0.003",
            facecolor="#ffcdd2", edgecolor="none", alpha=0.5,
            transform=ax.transAxes)
        ax.add_patch(rect)
    ax.text(0.04, y, line, fontproperties=fp, color=c, va="top",
            transform=ax.transAxes)
    y -= 0.045

# ── right panel: Steered ────────────────────────────────────────────────────
st_box = mpatches.FancyBboxPatch(
    (0.52, 0.06), 0.46, 0.72, boxstyle="round,pad=0.02",
    facecolor="#f1f8e9", edgecolor="#2e7d32", linewidth=2,
    transform=ax.transAxes)
ax.add_patch(st_box)
ax.text(0.54, 0.76, "Steered  ✓  CORRECT",
        fontproperties=FP_B, color="#1b5e20", va="top", transform=ax.transAxes)

y = 0.70
for i, (line, bold) in enumerate(steered_lines):
    fp = FP_B if bold else FP_M
    c = "#1b5e20" if i in st_highlight else "#333"
    if i in st_highlight:
        rect = mpatches.FancyBboxPatch(
            (0.53, y - 0.005), 0.44, 0.038, boxstyle="round,pad=0.003",
            facecolor="#c8e6c9", edgecolor="none", alpha=0.5,
            transform=ax.transAxes)
        ax.add_patch(rect)
    ax.text(0.54, y, line, fontproperties=fp, color=c, va="top",
            transform=ax.transAxes)
    y -= 0.045

# ── center arrow ────────────────────────────────────────────────────────────
ax.annotate("", xy=(0.515, 0.45), xytext=(0.495, 0.45),
            arrowprops=dict(arrowstyle="-|>", color="#1565c0", lw=3,
                            mutation_scale=22),
            xycoords="axes fraction")

# ── bottom annotation ──────────────────────────────────────────────────────
ax.text(0.5, 0.02,
        'Key change: Baseline misinterprets "3 times fewer" as multiplication (3×21=63);\n'
        'Steered correctly applies division (21÷3=7), producing denser steps with explicit equations.',
        fontproperties=FP_S, ha="center", va="bottom", color="#555",
        transform=ax.transAxes, linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#e3f2fd", edgecolor="#90caf9", alpha=0.7))

out_path = OUT / "flipped_531_diff.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved -> {out_path}")
plt.close(fig)
