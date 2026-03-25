#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json
import numpy as np
import matplotlib.pyplot as plt

# ====== CONFIG ======
SAVE_DIR = "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_less_tokens_3B_lib_manual_same_lib_1000samples copy/prm_results"
RESULTS_JSON = os.path.join(SAVE_DIR, "results_merged.json")
OUT_DIR = os.path.join(SAVE_DIR, "special")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_ACC = os.path.join(OUT_DIR, "acc_vs_lambda.png")
OUT_NLL = os.path.join(OUT_DIR, "nll_vs_lambda.png")

# ====== helpers ======
def _sorted_lams(layer_dict):
    # keys may be strings like "0", "1.0", "-5"
    lams = []
    for k in layer_dict.keys():
        try:
            lams.append(float(k))
        except Exception:
            pass
    if lams:
        return sorted(lams)

    # 如果没有任何数值 lambda，但存在 BASELINE，就当成 lambda=0.0 画一个点
    if "BASELINE" in layer_dict:
        return [0.0]

    return []

def _get_entry(layer_dict, lam_float):
    # robustly fetch by exact string formatting
    # try common representations
    for key in (str(lam_float), f"{lam_float}", f"{lam_float:.1f}", f"{lam_float:.2f}", f"{lam_float:.3f}"):
        if key in layer_dict:
            return layer_dict[key]

    # 特判 BASELINE：如果我们把它映射到 lambda=0.0
    if lam_float == 0.0 and "BASELINE" in layer_dict:
        return layer_dict["BASELINE"]
    # fallback: search by float-equality
    for k, v in layer_dict.items():
        try:
            if float(k) == lam_float:
                return v
        except Exception:
            continue
    raise KeyError(f"Cannot find entry for lam={lam_float} in keys={list(layer_dict.keys())[:10]}...")

def _compute_acc(entry):
    return float(np.mean(np.array(entry["Y"], dtype=float)))

def _compute_nll(entry):
    ppl = entry.get("ppl")
    # 有些 entry 没有 ppl，直接返回 None 让上层跳过
    if ppl is None:
        return None

    # ppl may be a scalar or list
    if isinstance(ppl, (list, tuple, np.ndarray)):
        if len(ppl) == 0:
            return None
        ppl_mean = float(np.mean(np.array(ppl, dtype=float)))
    else:
        ppl_mean = float(ppl)

    # NLL = log(PPL)
    return float(np.log(ppl_mean))

def _setup_rcparams():
    plt.rcParams.update({
        "font.size": 20,
        "axes.labelsize": 20,
        "legend.fontsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    })

def plot_metric_vs_lambda(raw, metric_name, y_label, out_path):
    """
    metric_name: "acc" or "nll"
    Each line is (model, layer). x=lam, y=metric.
    """
    _setup_rcparams()

    fig, ax = plt.subplots(figsize=(9, 6))

    # colors by model, markers by layer (cycled)
    all_models = list(raw.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(all_models))))
    model2color = {m: colors[i] for i, m in enumerate(all_models)}

    markers = ["o", "s", "^", "D", "P", "X", "<", ">", "v", "*"]

    handles, labels = [], []
    seen = set()

    for mi, (model_name, model_data) in enumerate(raw.items()):
        short_m = model_name.split("/")[-1]

        # layers could be ints or strings
        layers = list(model_data.keys())
        # try numeric sort if possible
        try:
            layers = sorted(layers, key=lambda x: int(x))
        except Exception:
            layers = sorted(layers, key=lambda x: str(x))

        for li, L in enumerate(layers):
            layer_dict = model_data[L]
            lams = _sorted_lams(layer_dict)
            if not lams:
                continue

            xs, ys = [], []
            for lam in lams:
                entry = _get_entry(layer_dict, lam)
                xs.append(lam)
                if metric_name == "acc":
                    ys.append(_compute_acc(entry))
                elif metric_name == "nll":
                    nll = _compute_nll(entry)
                    # 若该 entry 没有 ppl，则跳过这个点
                    if nll is None:
                        continue
                    ys.append(nll)
                else:
                    raise ValueError(metric_name)

            label = f"{short_m}-L{L}"
            # 去重 legend（如果你 layer 很多可以只保留 model 级别：把 label 改成 short_m）
            if label in seen:
                label = None
            else:
                seen.add(label)

            line, = ax.plot(
                xs, ys,
                marker=markers[li % len(markers)],
                linewidth=2.0,
                markersize=7,
                color=model2color[model_name],
                alpha=0.9,
                label=label
            )
            if label is not None:
                handles.append(line)
                labels.append(label)

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)

    # legend on top
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=min(3, max(1, len(labels))),
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)


def main():
    with open(RESULTS_JSON, "r") as f:
        raw = json.load(f)

    # 始终可以画 Accuracy vs lambda
    plot_metric_vs_lambda(raw, metric_name="acc", y_label="Accuracy", out_path=OUT_ACC)

    # 只有在 entry 里真的有 ppl 字段时才尝试画 NLL
    try:
        first_model = next(iter(raw.values()))
        first_layer = next(iter(first_model.values()))
        first_entry = next(iter(first_layer.values()))
        has_ppl = "ppl" in first_entry
    except StopIteration:
        has_ppl = False

    if has_ppl:
        plot_metric_vs_lambda(raw, metric_name="nll", y_label="Negative Log-Likelihood (NLL)", out_path=OUT_NLL)
    else:
        print("[Skip] No 'ppl' field found in entries; skip NLL plot.")

if __name__ == "__main__":
    main()