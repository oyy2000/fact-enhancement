#!/usr/bin/env python3
"""Plot NLL = log(PPL) heatmap for the FactSteer grid (L17, L31, L35).

Reads directly from chunk files (not the 182M-line merged JSON).
"""
import json, math, re
from pathlib import Path
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement")
COPY_PRM = (
    BASE
    / "lm-evaluation-harness/lm_eval/models"
    / "eval_grid_less_tokens_3B_lib_manual_same_lib_1000samples copy"
    / "prm_results copy"
)
OUT_DIR = COPY_PRM / "all"
OUT_DIR.mkdir(exist_ok=True)

# ── Load all chunks ──────────────────────────────────────────────────────────
data = {}  # data[layer][lam_str] = {"ppl": [...], "nll": [...]}
for cf in sorted(COPY_PRM.glob("results_chunk_*.json")):
    blob = json.load(open(cf))
    for model in blob:
        for L in blob[model]:
            for lam in blob[model][L]:
                entry = blob[model][L][lam]
                ppl_list = entry.get("ppl", [])
                if not ppl_list:
                    continue
                nll_list = entry.get("nll") or [math.log(p) for p in ppl_list if p > 0]
                data.setdefault(L, {})[lam] = {
                    "mean_ppl": np.mean(ppl_list),
                    "mean_nll": np.mean(nll_list),
                    "ppl": ppl_list,
                    "nll": nll_list,
                }

print(f"Loaded layers: {sorted(data.keys())}")
for L in sorted(data.keys()):
    print(f"  {L}: {len(data[L])} lambdas")


def lam_to_float(s):
    if s == "BASELINE":
        return 0.0
    neg = s.startswith("lam-")
    body = s.replace("lam-", "").replace("lam", "")
    val = float(body.replace("p", "."))
    return -val if neg else val


# ── Line plot: NLL vs λ per layer ────────────────────────────────────────────
LAYER_COLORS = {
    "L8": "#7b1fa2", "L10": "#c62828", "L17": "#1a237e",
    "L24": "#00838f", "L31": "#e65100", "L35": "#2e7d32",
}
PLOT_LAYERS = ["L17", "L31", "L35"]

fig, ax = plt.subplots(figsize=(10, 6))
for L in sorted(data.keys()):
    if L not in PLOT_LAYERS:
        continue
    lams_sorted = sorted(data[L].items(), key=lambda kv: lam_to_float(kv[0]))
    xs = [lam_to_float(lam) for lam, _ in lams_sorted]
    ys = [v["mean_nll"] for _, v in lams_sorted]
    ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=4,
            color=LAYER_COLORS.get(L, "#333"), label=L)

ax.set_xlabel(r"$\lambda$", fontsize=12)
ax.set_ylabel("NLL = log(PPL)", fontsize=12)
ax.set_title("FactSteer: NLL vs Steering Strength", fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
fig.tight_layout()
outp = OUT_DIR / "nll.png"
fig.savefig(outp, dpi=200)
plt.close()
print(f"wrote {outp}")

# ── Also plot PPL for comparison ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
for L in sorted(data.keys()):
    if L not in PLOT_LAYERS:
        continue
    lams_sorted = sorted(data[L].items(), key=lambda kv: lam_to_float(kv[0]))
    xs = [lam_to_float(lam) for lam, _ in lams_sorted]
    ys = [v["mean_ppl"] for _, v in lams_sorted]
    ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=4,
            color=LAYER_COLORS.get(L, "#333"), label=L)

ax.set_xlabel(r"$\lambda$", fontsize=12)
ax.set_ylabel("Perplexity (PPL)", fontsize=12)
ax.set_title("FactSteer: PPL vs Steering Strength", fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
fig.tight_layout()
outp = OUT_DIR / "ppl.png"
fig.savefig(outp, dpi=200)
plt.close()
print(f"wrote {outp}")

print("ALL_DONE")
