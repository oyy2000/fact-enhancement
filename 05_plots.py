import json
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
from transformers import AutoTokenizer


# ============================================================
# CONFIG
# ============================================================

LAM_MARKERS = ["o", "s", "^", "D", "P", "X", "<", ">", "v"]

def get_lambda_marker_map(lambdas):
    lambdas = sorted(lambdas, key=lam_to_float)
    return {lam: LAM_MARKERS[i % len(LAM_MARKERS)] for i, lam in enumerate(lambdas)}

THRESHOLD = 0.72
def get_layer_color_map(layers):
    colors = plt.cm.tab10(np.linspace(0, 1, len(layers)))
    return {L: colors[i] for i, L in enumerate(layers)}

SAVE_DIR = "prm_out_qwen_family_14btos"
MODEL_FILES = [
    f"/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/{SAVE_DIR}/results_merged.json",
    # 你可以继续添加其它模型族
    # "/path/to/prm_out_llama/results_merged.json",
]

SAVE_ROOT = f"/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/{SAVE_DIR}"
os.makedirs(SAVE_ROOT, exist_ok=True)
os.makedirs(f"{SAVE_ROOT}/correct_wrong", exist_ok=True)
os.makedirs(f"{SAVE_ROOT}/scatter", exist_ok=True)
os.makedirs(f"{SAVE_ROOT}/special", exist_ok=True)

MODEL_MARKERS = ["o", "s", "^", "D", "P", "X", "<", ">", "v"]
WRONG_MARKERS = ["x", "X", "v", "p", "*", "d", "h", "H", "+"]

MODEL_LINESTYLES = ["-", "--", "-.", ":"]
LAYER_ALPHA = [1.0, 0.7, 0.5, 0.2]


LAYER_MARKERS = ["o", "s", "^", "D", "P", "X"]

# ============================================================
# Helper Functions
# ============================================================

def detect_layers(model_data):
    return list(model_data.keys())


def detect_lambdas(model_data, L):
    return list(model_data[L].keys())


def lam_to_float(lam):
    if lam == "BASELINE":
        return 0.0
    return float(lam[3:].replace("p", "."))


def compute_prefix_first_error(step_scores, thr):
    prefix_list = []
    fe_list = []
    for scores in step_scores:
        prefix = 0
        fe = None
        for i, s in enumerate(scores):
            if s >= thr:
                prefix += 1
            else:
                fe = i + 1
                break
        if fe is None:
            fe = len(scores) + 1
        prefix_list.append(prefix)
        fe_list.append(fe)
    return np.array(prefix_list), np.array(fe_list)


def compute_corr(x, y):
    if len(x) < 3 or np.std(x) == 0:
        return None
    try:
        return pearsonr(x, y)[0]
    except:
        return None

def extract_metrics(entry, thr, model_name=None):
    step_scores = entry["step_scores"]          # List[List[float]]
    step_token_len = entry["step_token_len"]    # List[List[int]]
    Y = np.array(entry["Y"])

    # ===== prefix / first error =====
    prefix, first_err = compute_prefix_first_error(step_scores, thr)

    # ===== step-level metrics =====
    avg_scores = np.array([np.mean(s) for s in step_scores])
    avg_steps  = np.array([len(s) for s in step_scores])

    # ===== token-based metrics (FROM JSON, NO tokenizer) =====
    avg_tokens_per_step = np.array([
        np.mean(lens) for lens in step_token_len
    ])

    avg_total_tokens = np.array([
        np.sum(lens) for lens in step_token_len
    ])
    # ===== total error steps (count of steps with score < thr) =====
    total_error_steps = np.array([
        np.sum(np.array(s) < thr) for s in step_scores
    ])

    # ===== hard first-step flag =====
    F_hard = np.array([
        1 if s[0] >= thr else 0 for s in step_scores
    ])
    error_step_ratio = np.array([
        np.sum(np.array(s) < thr) / max(len(s), 1) for s in step_scores
    ])

    return {
        "prefix": prefix,
        "first_error": first_err,
        "avg_scores": avg_scores,
        "avg_steps": avg_steps,
        "total_error_steps": total_error_steps,   # ⭐ 新增
        "error_step_ratio": error_step_ratio,
        "avg_tokens_per_step": avg_tokens_per_step,
        "avg_total_tokens": avg_total_tokens,
        "F_hard": F_hard,
        "Y": Y
    }


# ============================================================
# Load All Models (each JSON may contain multiple models)
# ============================================================

print("Loading all models...")
model_results = {}

for file in MODEL_FILES:
    raw = json.load(open(file))
    for model_name, model_data in raw.items():
        print(model_name)
        model_results[model_name] = model_data
        print("Loaded model:", model_name)

all_models = list(model_results.keys())
print("\nTotal models:", len(all_models))


# ============================================================
# Assign Color, Marker, Linestyle Per Model
# ============================================================

color_list = plt.cm.tab20(np.linspace(0, 1, len(all_models)))
MODEL_COLOR_MAP = {
    model_name: color_list[i]
    for i, model_name in enumerate(all_models)
}

MODEL_MARKER_MAP = {
    model_name: MODEL_MARKERS[i % len(MODEL_MARKERS)]
    for i, model_name in enumerate(all_models)
}

MODEL_LINESTYLE_MAP = {
    model_name: MODEL_LINESTYLES[i % len(MODEL_LINESTYLES)]
    for i, model_name in enumerate(all_models)
}


# ============================================================
#  A. Correct vs Wrong Plots
# ============================================================
def plot_correct_wrong():
    metrics = {
        "prefix": "Avg Prefix Length",
        "first_error": "Avg First Error Step",
        "avg_scores": "Avg Step Correctness",
        "avg_steps": "Avg Total Steps",
        "avg_tokens_per_step": "Avg Tokens per Step",
        "avg_total_tokens": "Avg Total Tokens",
        "total_error_steps": "Total Error Steps",
        "error_step_ratio": "Error Step Ratio"
    }

    WRONG_MARKERS = ["x", "X", "v", "p", "*", "d", "h", "H", "+"]

    outdir = f"{SAVE_ROOT}/correct_wrong"

    for metric_key, metric_name in metrics.items():
        plt.figure(figsize=(10,6))

        # enumerate 确保模型有 index i（用于 wrong marker）
        for i, (model_name, model_data) in enumerate(model_results.items()):
            
            # Wrong marker 由模型 index i 决定
            wrong_marker = WRONG_MARKERS[i % len(WRONG_MARKERS)]

            layers = detect_layers(model_data)
            alpha_map = {L: LAYER_ALPHA[j % len(LAYER_ALPHA)] for j, L in enumerate(layers)}

            for L in layers:
                lambdas = detect_lambdas(model_data, L)

                lam_vals = []
                corr_vals = []
                wrong_vals = []

                for lam in lambdas:
                    entry = model_data[L][lam]
                    M = extract_metrics(entry, THRESHOLD, model_name)
                    Y = M["Y"]

                    correct_mean = np.mean(M[metric_key][Y == 1])
                    wrong_mean   = np.mean(M[metric_key][Y == 0])

                    lam_num = lam_to_float(lam)

                    lam_vals.append(lam_num)
                    corr_vals.append(correct_mean)
                    wrong_vals.append(wrong_mean)

                # sorting by lambda
                idx = np.argsort(lam_vals)
                lam_vals = np.array(lam_vals)[idx]
                corr_vals = np.array(corr_vals)[idx]
                wrong_vals = np.array(wrong_vals)[idx]

                # Correct curve
                plt.plot(
                    lam_vals, corr_vals,
                    color=MODEL_COLOR_MAP[model_name],
                    linestyle=MODEL_LINESTYLE_MAP[model_name],
                    marker=MODEL_MARKER_MAP[model_name],     # correct marker (模型)
                    markersize=7,
                    alpha=alpha_map[L],
                    label=f"{model_name}-{L} (Correct)"
                )

                # Wrong curve
                plt.plot(
                    lam_vals, wrong_vals,
                    color=MODEL_COLOR_MAP[model_name],
                    linestyle=MODEL_LINESTYLE_MAP[model_name],
                    marker=wrong_marker,                     # wrong marker (不同样式)
                    markersize=8,
                    alpha=alpha_map[L],
                    label=f"{model_name}-{L} (Wrong)"
                )

        plt.title(f"Correct vs Wrong — {metric_name}")
        plt.xlabel("λ")
        plt.ylabel(metric_name)
        plt.grid(alpha=0.3)

        # dedupe legend
        handles, labels = plt.gca().get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        plt.legend(uniq.values(), uniq.keys(), bbox_to_anchor=(1.05,1), loc="upper left")

        plt.tight_layout()
        out_path = f"{outdir}/{metric_key}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print("Saved:", out_path)


# ============================================================
#  D. All-samples plots (no Correct/Wrong split)
# ============================================================
def plot_all():
    metrics = {
        "prefix": "Avg Prefix Length",
        "first_error": "Avg First Error Step",
        "avg_scores": "Avg Step Correctness",
        "avg_steps": "Avg Total Steps",
        "avg_tokens_per_step": "Avg Tokens per Step",
        "avg_total_tokens": "Avg Total Tokens",
        "acc": "Accuracy",   # ⭐ 新增
        "total_error_steps": "Total Error Steps",
        "error_step_ratio": "Error Step Ratio"
    }

    outdir = f"{SAVE_ROOT}/all"
    os.makedirs(outdir, exist_ok=True)

    for metric_key, metric_name in metrics.items():
        plt.figure(figsize=(10,6))
        ax = plt.gca()
        ax_acc = ax.twinx()   # ⭐ 右轴：accuracy

        for model_name, model_data in model_results.items():
            layers = detect_layers(model_data)
            alpha_map = {
                L: LAYER_ALPHA[i % len(LAYER_ALPHA)]
                for i, L in enumerate(layers)
            }

            for L in layers:
                lambdas = detect_lambdas(model_data, L)

                lam_vals = []
                mean_vals = []

                for lam in lambdas:
                    entry = model_data[L][lam]
                    M = extract_metrics(entry, THRESHOLD, model_name)
                    Y = M["Y"]        # 0/1 correctness
                    acc = np.mean(Y) # ⭐ 这就是 accuracy

                    lam_num = lam_to_float(lam)
                    if metric_key == "acc":
                        mean_val = np.mean(M["Y"])      # ⭐ accuracy
                    else:
                        mean_val = np.mean(M[metric_key])

                    lam_vals.append(lam_num)
                    mean_vals.append(mean_val)

                # sort by λ
                idx = np.argsort(lam_vals)
                lam_vals = np.array(lam_vals)[idx]
                mean_vals = np.array(mean_vals)[idx]

                plt.plot(
                    lam_vals,
                    mean_vals,
                    color=MODEL_COLOR_MAP[model_name],
                    linestyle=MODEL_LINESTYLE_MAP[model_name],
                    marker=MODEL_MARKER_MAP[model_name],
                    markersize=6,
                    alpha=alpha_map[L],
                    label=f"{model_name}-{L}"
                )

        plt.title(f"All Samples — {metric_name}")
        plt.xlabel("λ")
        plt.ylabel(metric_name)
        plt.grid(alpha=0.3)

        # dedupe legend
        handles, labels = plt.gca().get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        plt.legend(
            uniq.values(),
            uniq.keys(),
            bbox_to_anchor=(1.05,1),
            loc="upper left"
        )

        # if metric_key == "acc":
            # plt.ylim(0, 1.05)

        plt.tight_layout()
        out_path = f"{outdir}/{metric_key}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print("Saved:", out_path)



# # ============================================================
# #  D. All-samples plots (no Correct/Wrong split)
# # ============================================================
# def plot_all():
#     metrics = {
#         "prefix": "Avg Prefix Length",
#         "first_error": "Avg First Error Step",
#         "avg_scores": "Avg Step Correctness",
#         "avg_steps": "Avg Total Steps",
#         "avg_tokens_per_step": "Avg Tokens per Step",
#         "avg_total_tokens": "Avg Total Tokens",
#     }

#     outdir = f"{SAVE_ROOT}/all"
#     os.makedirs(outdir, exist_ok=True)

#     for metric_key, metric_name in metrics.items():
#         plt.figure(figsize=(10,6))
#         ax = plt.gca()
#         ax_acc = ax.twinx()   # ⭐ 右轴：accuracy

#         for model_name, model_data in model_results.items():
#             layers = detect_layers(model_data)
#             alpha_map = {
#                 L: LAYER_ALPHA[i % len(LAYER_ALPHA)]
#                 for i, L in enumerate(layers)
#             }

#             for L in layers:
#                 lambdas = detect_lambdas(model_data, L)

#                 lam_vals = []
#                 mean_vals = []
#                 acc_vals = []
#                 for lam in lambdas:
#                     entry = model_data[L][lam]
#                     M = extract_metrics(entry, THRESHOLD, model_name)
#                     acc = np.mean(M["Y"]) # ⭐ 这就是 accuracy

#                     lam_num = lam_to_float(lam)
#                     mean_val = np.mean(M[metric_key])   # ⭐ 不区分 Y

#                     lam_vals.append(lam_num)
#                     mean_vals.append(mean_val)
#                     acc_vals.append(acc)           # ⭐

#                 # sort by λ
#                 idx = np.argsort(lam_vals)
#                 lam_vals = np.array(lam_vals)[idx]
#                 mean_vals = np.array(mean_vals)[idx]
#                 acc_vals = np.array(acc_vals)[idx]   # ⭐

#                 ax.plot(
#                     lam_vals,
#                     mean_vals,
#                     color=MODEL_COLOR_MAP[model_name],
#                     linestyle=MODEL_LINESTYLE_MAP[model_name],
#                     marker=MODEL_MARKER_MAP[model_name],
#                     markersize=6,
#                     alpha=alpha_map[L],
#                     label=f"{model_name}-{L}"
#                 )

#                 # ⭐ Accuracy（右轴）
#                 ax_acc.plot(
#                     lam_vals,
#                     acc_vals,
#                     color=MODEL_COLOR_MAP[model_name],
#                     linestyle="--",      # 用虚线区分
#                     alpha=alpha_map[L],
#                 )


#         plt.title(f"All Samples — {metric_name} (solid) & Accuracy (dashed)")
#         ax.set_xlabel("λ")
#         ax.set_ylabel(metric_name)
#         ax_acc.set_ylabel("Accuracy")
#         ax_acc.set_ylim(0, 1.05)   # ⭐ acc 语义稳定

#         # dedupe legend
#         handles, labels = plt.gca().get_legend_handles_labels()
#         uniq = dict(zip(labels, handles))
#         plt.legend(
#             uniq.values(),
#             uniq.keys(),
#             bbox_to_anchor=(1.05,1),
#             loc="upper left"
#         )

#         plt.tight_layout()
#         out_path = f"{outdir}/{metric_key}.png"
#         plt.savefig(out_path, dpi=300)
#         plt.close()
#         print("Saved:", out_path)



def lam_to_size(lam, base=80, scale=40):
    return base + scale * abs(lam)

# ============================================================
#  B. Scatter plots
# ============================================================
def plot_scatter():

    metrics = {
        "avg_prefix": "Avg Prefix Length",
        "avg_first_error": "Avg First Error Step",
        "avg_steps": "Avg Total Steps",
        "avg_total_tokens": "Avg Total Tokens",
    }

    outdir = f"{SAVE_ROOT}/scatter"

    for model_name, model_data in model_results.items():
        layers = detect_layers(model_data)
        layer_color_map = get_layer_color_map(layers)

        for metric, metric_name in metrics.items():

            plt.figure(figsize=(10,6))

            for L in layers:
                lambdas = detect_lambdas(model_data, L)
                lam_marker_map = get_lambda_marker_map(lambdas)
                for lam in lambdas:
                    entry = model_data[L][lam]
                    M = extract_metrics(entry, THRESHOLD, model_name)
                    Y = M["Y"]
                    acc = np.mean(Y)

                    prefix = M["prefix"]
                    steps = M["avg_steps"]
                    first = M["first_error"]
                    avg_total_tokens = M["avg_total_tokens"]

                    if metric == "avg_prefix":
                        x = np.mean(prefix)
                    elif metric == "avg_first_error":
                        x = np.mean(first)
                    elif metric == "avg_steps":
                        x = np.mean(steps)
                    elif metric == "avg_total_tokens":
                        x = np.mean(avg_total_tokens)

                    plt.scatter(
                        x, acc,
                        color=layer_color_map[L],          # layer → color
                        marker=lam_marker_map[lam],        # λ → shape ✅
                        s=80,                              # 固定大小
                        alpha=0.85,
                        edgecolors="k",
                        linewidths=0.5,
                        label = f"Layer {L}" if lam == lambdas[0] else None
                    )


            # ---- figure-level decoration ----
            plt.xlabel(metric_name)
            plt.ylabel("Accuracy")
            plt.title(f"{model_name}: {metric_name} vs Accuracy")
            plt.grid(alpha=0.3)
            
            ax = plt.gca()

            # ---- Layer legend ----
            handles, labels = ax.get_legend_handles_labels()
            uniq = dict(zip(labels, handles))

            layer_legend = ax.legend(
                uniq.values(), uniq.keys(),
                title="Layer",
                bbox_to_anchor=(1.05, 1),
                loc="upper left"
            )

            ax.add_artist(layer_legend)   # ⭐ 关键：手动固定住第一个 legend

                        # ---- lambda (marker) legend ----
            lambda_handles = []
            lambda_labels = []

            all_lambdas = sorted(
                {lam for L in layers for lam in detect_lambdas(model_data, L)},
                key=lam_to_float
            )
            lam_marker_map = get_lambda_marker_map(all_lambdas)

            for lam in all_lambdas:
                h = plt.Line2D(
                    [], [],
                    linestyle="",
                    marker=lam_marker_map[lam],
                    color="gray",
                    markersize=8,
                    label=lam
                )
                lambda_handles.append(h)
                lambda_labels.append(lam)

            ax.legend(
                lambda_handles, lambda_labels,
                title="λ",
                bbox_to_anchor=(1.05, 0),
                loc="lower left"
            )

            plt.tight_layout()
            out_path = f"{outdir}/{model_name}_{metric}.png"
            plt.savefig(out_path, dpi=300)
            plt.close()

            print("Saved:", out_path)




# ============================================================
#  C. Special plots
# ============================================================

def plot_first_step_vs_lambda():
    plt.figure(figsize=(10,6))

    for model_name, model_data in model_results.items():
        layers = detect_layers(model_data)

        lam_list = []
        fs_list = []

        for L in layers:
            for lam, entry in model_data[L].items():
                first_step = np.mean([s[0] for s in entry["step_scores"]])
                lam_num = lam_to_float(lam)

                lam_list.append(lam_num)
                fs_list.append(first_step)

        lam_list = np.array(lam_list)
        fs_list = np.array(fs_list)
        idx = np.argsort(lam_list)

        plt.plot(
            lam_list[idx],
            fs_list[idx],
            color=MODEL_COLOR_MAP[model_name],
            linestyle=MODEL_LINESTYLE_MAP[model_name],
            marker=MODEL_MARKER_MAP[model_name],
            label=model_name
        )

    plt.xlabel("λ")
    plt.ylabel("Avg First Step PRM Score")
    plt.title("First Step Score vs λ")
    plt.grid(alpha=0.3)
    plt.legend()

    out_path = f"{SAVE_ROOT}/special/first_step_vs_lambda.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)

def plot_avg_score_vs_acc():
    plt.figure(figsize=(10,6))

    seen_models = set()

    for model_name, model_data in model_results.items():
        layers = detect_layers(model_data)

        for L in layers:
            for lam, entry in model_data[L].items():
                avg_step_score = np.mean([np.mean(s) for s in entry["step_scores"]])
                acc = np.mean(entry["Y"])

                label = model_name if model_name not in seen_models else None
                seen_models.add(model_name)

                plt.scatter(
                    avg_step_score,
                    acc,
                    color=MODEL_COLOR_MAP[model_name],
                    marker=MODEL_MARKER_MAP[model_name],
                    s=140,
                    alpha=0.85,
                    label=label
                )

    plt.xlabel("Avg Step Score")
    plt.ylabel("Accuracy")
    plt.title("Avg Step Score vs Accuracy")
    plt.grid(alpha=0.3)
    plt.legend(title="Model")

    out_path = f"{SAVE_ROOT}/special/avg_step_score_vs_accuracy.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_avg_tokens_vs_acc():
    plt.figure(figsize=(10,6))
    seen_models = set()

    for model_name, model_data in model_results.items():
        layers = detect_layers(model_data)

        for L in layers:
            for lam, entry in model_data[L].items():
                avg_tokens = np.mean([
                    np.mean([len(t.split()) for t in steps])
                    for steps in entry["step_texts"]
                ])
                acc = np.mean(entry["Y"])
                label = model_name if model_name not in seen_models else None
                seen_models.add(model_name)

                plt.scatter(
                    avg_tokens,
                    acc,
                    color=MODEL_COLOR_MAP[model_name],
                    marker=MODEL_MARKER_MAP[model_name],
                    s=140,
                    alpha=0.85,
                    label=label
                )

    plt.xlabel("Avg Tokens per Step")
    plt.ylabel("Accuracy")
    plt.title("Avg Tokens vs Accuracy")
    plt.grid(alpha=0.3)
    plt.legend(title="Model")

    out_path = f"{SAVE_ROOT}/special/avg_tokens_vs_accuracy.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)


# ============================================================
#  E. Step-level correctness histogram
# ============================================================

def _safe_name(s: str) -> str:
    """Make a string safe for filenames."""
    return (
        s.replace("/", "_")
         .replace(" ", "_")
         .replace(":", "_")
         .replace("__", "_")
    )

def plot_step_correctness_hist(max_steps=10, bins=25, density=True):
    """
    For each (model, layer, lambda), plot histogram of PRM step score
    at step k (k=1..max_steps), separately for Correct vs Wrong samples.

    Saved as:
      SAVE_ROOT/step_hist/{model}_{L}_{lam}_step{k}.png
    """
    outdir = f"{SAVE_ROOT}/step_hist"
    os.makedirs(outdir, exist_ok=True)

    for model_name, model_data in model_results.items():
        layers = detect_layers(model_data)

        for L in layers:
            lambdas = detect_lambdas(model_data, L)

            for lam in lambdas:
                entry = model_data[L][lam]
                step_scores = entry["step_scores"]   # List[List[float]]
                Y = np.array(entry["Y"])            # 0/1

                # decide actual max step available for this config
                max_len = max((len(s) for s in step_scores), default=0)
                K = min(max_steps, max_len)

                for k in range(K):
                    # collect kth step scores for correct/wrong
                    corr_vals = [
                        step_scores[i][k]
                        for i in range(len(step_scores))
                        if Y[i] == 1 and len(step_scores[i]) > k
                    ]
                    wrong_vals = [
                        step_scores[i][k]
                        for i in range(len(step_scores))
                        if Y[i] == 0 and len(step_scores[i]) > k
                    ]

                    # if too few samples, skip
                    if len(corr_vals) < 2 and len(wrong_vals) < 2:
                        continue

                    plt.figure(figsize=(8, 5))

                    if len(corr_vals) > 0:
                        plt.hist(
                            corr_vals,
                            bins=bins,
                            density=density,
                            alpha=0.6,
                            label=f"Correct (n={len(corr_vals)})",
                            edgecolor="k",
                            linewidth=0.3,
                        )
                    if len(wrong_vals) > 0:
                        plt.hist(
                            wrong_vals,
                            bins=bins,
                            density=density,
                            alpha=0.6,
                            label=f"Wrong (n={len(wrong_vals)})",
                            edgecolor="k",
                            linewidth=0.3,
                        )

                    plt.axvline(THRESHOLD, linestyle="--", linewidth=1.2, label=f"thr={THRESHOLD}")

                    plt.title(f"Step-{k+1} Correctness Histogram\n{model_name} | L={L} | λ={lam}")
                    plt.xlabel("PRM Step Score")
                    plt.ylabel("Density" if density else "Count")
                    plt.grid(alpha=0.25)
                    plt.legend()

                    fname = f"{_safe_name(model_name)}_L{L}_{_safe_name(lam)}_step{k+1}.png"
                    out_path = os.path.join(outdir, fname)
                    plt.tight_layout()
                    plt.savefig(out_path, dpi=300)
                    plt.close()
                    print("Saved:", out_path)


def plot_step_correctness_hist_flatten(bins=30, density=True):
    """
    (Optional) For each (model, layer, lambda), flatten all step scores across all steps
    and draw one histogram (Correct vs Wrong).
    """
    outdir = f"{SAVE_ROOT}/step_hist_flat"
    os.makedirs(outdir, exist_ok=True)

    for model_name, model_data in model_results.items():
        layers = detect_layers(model_data)

        for L in layers:
            lambdas = detect_lambdas(model_data, L)

            for lam in lambdas:
                entry = model_data[L][lam]
                step_scores = entry["step_scores"]
                Y = np.array(entry["Y"])

                corr_vals = []
                wrong_vals = []
                for i, scores in enumerate(step_scores):
                    if Y[i] == 1:
                        corr_vals.extend(scores)
                    else:
                        wrong_vals.extend(scores)

                if len(corr_vals) < 2 and len(wrong_vals) < 2:
                    continue

                plt.figure(figsize=(8, 5))
                if len(corr_vals) > 0:
                    plt.hist(
                        corr_vals, bins=bins, density=density, alpha=0.6,
                        label=f"Correct (n={len(corr_vals)})",
                        edgecolor="k", linewidth=0.3,
                    )
                if len(wrong_vals) > 0:
                    plt.hist(
                        wrong_vals, bins=bins, density=density, alpha=0.6,
                        label=f"Wrong (n={len(wrong_vals)})",
                        edgecolor="k", linewidth=0.3,
                    )

                plt.axvline(THRESHOLD, linestyle="--", linewidth=1.2, label=f"thr={THRESHOLD}")
                plt.title(f"All-Steps Correctness Histogram\n{model_name} | L={L} | λ={lam}")
                plt.xlabel("PRM Step Score")
                plt.ylabel("Density" if density else "Count")
                plt.grid(alpha=0.25)
                plt.legend()

                fname = f"{_safe_name(model_name)}_L{L}_{_safe_name(lam)}_ALLSTEPS.png"
                out_path = os.path.join(outdir, fname)
                plt.tight_layout()
                plt.savefig(out_path, dpi=300)
                plt.close()
                print("Saved:", out_path)


def per_step_mean(step_scores, Y=None, max_steps=None):
    """
    Compute per-step mean PRM score across samples.
    - step_scores: List[List[float]]
    - Y: optional np.array of 0/1, if given we can filter outside before calling
    - max_steps: if None, use max length in step_scores
    Returns:
      means: np.array shape [K]
      counts: np.array shape [K]  (how many samples contribute at each step)
    """
    if len(step_scores) == 0:
        return np.array([]), np.array([])

    lens = np.array([len(s) for s in step_scores], dtype=int)
    if max_steps is None:
        K = int(lens.max()) if lens.size > 0 else 0
    else:
        K = int(min(max_steps, lens.max())) if lens.size > 0 else int(max_steps)

    means = np.full(K, np.nan, dtype=float)
    counts = np.zeros(K, dtype=int)

    for k in range(K):
        vals = [s[k] for s in step_scores if len(s) > k]
        counts[k] = len(vals)
        if counts[k] > 0:
            means[k] = float(np.mean(vals))
    return means, counts


def plot_per_step_avg_correctness(max_steps=15):
    """
    For each (model, layer, lambda), plot mean step correctness vs step index.
    Also supports Correct-only and Wrong-only curves on the same plot.
    """
    outdir = f"{SAVE_ROOT}/per_step_avg"
    os.makedirs(outdir, exist_ok=True)

    for model_name, model_data in model_results.items():
        layers = detect_layers(model_data)

        for L in layers:
            lambdas = detect_lambdas(model_data, L)

            for lam in lambdas:
                entry = model_data[L][lam]
                step_scores_all = entry["step_scores"]
                Y = np.array(entry["Y"])

                # --- split ---
                step_scores_corr = [step_scores_all[i] for i in range(len(step_scores_all)) if Y[i] == 1]
                step_scores_wrong = [step_scores_all[i] for i in range(len(step_scores_all)) if Y[i] == 0]

                mean_all, n_all = per_step_mean(step_scores_all, max_steps=max_steps)
                mean_corr, n_corr = per_step_mean(step_scores_corr, max_steps=max_steps)
                mean_wrong, n_wrong = per_step_mean(step_scores_wrong, max_steps=max_steps)

                if len(mean_all) == 0:
                    continue

                steps = np.arange(1, len(mean_all) + 1)

                plt.figure(figsize=(9, 5))
                plt.plot(steps, mean_all, marker="o", linewidth=2, label=f"All (n@1={n_all[0]})")
                if len(mean_corr) > 0:
                    plt.plot(steps, mean_corr, marker="s", linewidth=2, label=f"Correct (n@1={n_corr[0]})")
                if len(mean_wrong) > 0:
                    plt.plot(steps, mean_wrong, marker="x", linewidth=2, label=f"Wrong (n@1={n_wrong[0]})")

                plt.axhline(THRESHOLD, linestyle="--", linewidth=1.2, label=f"thr={THRESHOLD}")
                plt.xlabel("Step k")
                plt.ylabel("Avg PRM Step Score (across samples)")
                plt.title(f"Per-step Avg Correctness\n{model_name} | L={L} | λ={lam}")
                plt.grid(alpha=0.25)
                plt.legend()

                # 可选：把每一步有效样本数标出来（越往后会掉）
                # 你也可以改成 log 或者只画 n_all
                # plt.twinx().plot(steps, n_all, linestyle="--", marker=".", label="n_all")

                fname = f"{_safe_name(model_name)}_L{L}_{_safe_name(lam)}_per_step_avg.png"
                out_path = os.path.join(outdir, fname)
                plt.tight_layout()
                plt.savefig(out_path, dpi=300)
                plt.close()
                print("Saved:", out_path)

# ============================================================
# Run All
# ============================================================

# print("\n=== Generating Correct-vs-Wrong plots ===")
# plot_correct_wrong()

# print("\n=== Generating All-sample plots ===")
# plot_all()

# print("\n=== Generating Scatter plots ===")
# plot_scatter()

# print("\n=== Generating Step-level correctness histograms ===")
# plot_step_correctness_hist(max_steps=10, bins=25, density=True)

# # （可选）把所有 steps 拉平做一个整体分布直方图
# print("\n=== Generating Flattened step correctness histograms ===")
# plot_step_correctness_hist_flatten(bins=30, density=True)

# print("\n=== Generating Special plots ===")
# plot_first_step_vs_lambda()
# plot_avg_score_vs_acc()
# plot_avg_tokens_vs_acc()
print("\n=== Generating Per-step average correctness curves ===")
plot_per_step_avg_correctness(max_steps=15)


print("\n🎉 FINAL VERSION COMPLETE!  Saved in:", SAVE_ROOT)
