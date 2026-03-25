#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json
import numpy as np
import matplotlib.pyplot as plt

# ====== CONFIG ======
SAVE_DIR = "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_qwen_family/gsm8k_cot_zeroshot/prm_out_qwen_family_calibrated_prm_calibrated_split_calibrated_1"
RESULTS_JSON = os.path.join(SAVE_DIR, "results_merged.json")
OUT_PATH = os.path.join(SAVE_DIR, "special", "steps_and_tokens_per_step_vs_accuracy.png")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# ====== tiny helpers ======
def detect_layers(model_data):            # model_data: {layer: {...}}
    return list(model_data.keys())

def detect_lambdas(model_data, L):        # model_data[layer]: {lam: entry}
    return list(model_data[L].keys())

# ====== main plot ======
def main():
    raw = json.load(open(RESULTS_JSON))

    all_models = list(raw.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_models)))
    markers = ["o", "s", "^", "D", "P", "X", "<", ">", "v"]
    MODEL_COLOR = {m: colors[i] for i, m in enumerate(all_models)}
    MODEL_MARK  = {m: markers[i % len(markers)] for i, m in enumerate(all_models)}

    # bigger fonts
    plt.rcParams.update({
        "font.size": 20,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "legend.fontsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 13), sharey=True)

    seen = set()
    handles, labels = [], []

    def add_point(ax, x, y, model):
        short = model.split("/")[-1]
        label = short if short not in seen else None
        sc = ax.scatter(x, y, color=MODEL_COLOR[model], marker=MODEL_MARK[model],
                        s=160, alpha=0.85, label=label)
        if label is not None:
            seen.add(label)
            handles.append(sc)
            labels.append(label)

    for model_name, model_data in raw.items():
        for L in detect_layers(model_data):
            for lam in detect_lambdas(model_data, L):
                entry = model_data[L][lam]
                Y = np.array(entry["Y"], dtype=float)
                acc = float(np.mean(Y))

                step_token_len = entry["step_token_len"]  # List[List[int]]

                # top: avg steps
                num_steps = [len(s) for s in step_token_len]
                avg_steps = float(np.mean(num_steps))
                add_point(ax1, avg_steps, acc, model_name)

                # bottom: avg tokens per step
                tokens_per_step = [np.mean(s) for s in step_token_len]
                avg_tokens_per_step = float(np.mean(tokens_per_step))
                add_point(ax2, avg_tokens_per_step, acc, model_name)

    # ax1.set_title("Avg Step  vs Accuracy")
    ax1.set_xlabel("Avg Number of Steps")
    ax1.set_ylabel("Accuracy")
    ax1.grid(alpha=0.3)

    # ax2.set_title("Avg Tokens per Step vs Accuracy")
    ax2.set_xlabel("Avg Tokens per Step")
    ax2.set_ylabel("Accuracy")
    ax2.grid(alpha=0.3)

    fig.legend(handles, labels, loc="upper center",
               ncol=min(2, len(labels)), frameon=False,
               bbox_to_anchor=(0.5,1.05))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUT_PATH, dpi=400, bbox_inches="tight")
    plt.close()
    print("Saved:", OUT_PATH)

if __name__ == "__main__":
    main()
