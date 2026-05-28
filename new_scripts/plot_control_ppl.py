#!/usr/bin/env python3
"""Quick standalone plot from control_ppl_grid_eval.json."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path

DOCS = Path("/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/documents")

with open(DOCS / "control_ppl_grid_eval.json") as f:
    series = json.load(f)

EXPERIMENTS = ("e0", "e1", "e2", "e3", "e3.2")
EXP_COLORS = {"e0": "#1a237e", "e1": "#3949ab", "e2": "#5c6bc0", "e3": "#7986cb", "e3.2": "#9fa8da"}
EXP_LABELS = {
    "e0": "E0 (fact-enhanced)",
    "e1": "E1 (random-token control)",
    "e2": "E2 (shuffled-fact control)",
    "e3": "E3 (reversed-fact control)",
    "e3.2": "E3.2 (paraphrase control)",
}

lam_sorted = [round(-5.0 + i * 0.5, 1) for i in range(21)]

plt.figure(figsize=(9, 5.5))
for exp in EXPERIMENTS:
    if exp not in series:
        continue
    ys = [series[exp].get(str(l), float("nan")) for l in lam_sorted]
    plt.plot(lam_sorted, ys, marker="o", linewidth=1.5, markersize=4,
             color=EXP_COLORS.get(exp, "#333"), label=EXP_LABELS.get(exp, exp))

plt.xlabel(r"$\lambda$")
plt.ylabel("Perplexity (PPL)")
plt.xlim(-5.3, 5.3)
plt.gca().xaxis.set_major_locator(MultipleLocator(1.0))
plt.grid(True, alpha=0.35)
plt.legend(loc="best", fontsize=8)
plt.title("Impact of Steering Strength on Token-Level PPL (Control Experiments, L6)")
plt.figtext(0.5, 0.02,
    "Response-only PPL = exp(mean CE over all response tokens). "
    "Same as ModelEvaluator.compute_ppl_and_rank on generated_text. "
    "Qwen2.5-3B-Instruct base model, GSM8K CoT.",
    ha="center", fontsize=7)
plt.tight_layout(rect=[0, 0.07, 1, 1])
outp = DOCS / "control_ppl_vs_lambda_L6.png"
plt.savefig(outp, dpi=200)
plt.close()
print(f"wrote {outp}")
