import json
import matplotlib.pyplot as plt
import numpy as np
import os

# ====================== Load Results ======================
# json_path = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/prm_out/results_merged.json"
PREFIX = "qwen_family"
json_path = f"/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/prm_out_{PREFIX}/results_merged.json"

with open(json_path, "r") as f:
    results = json.load(f)
THRESHOLD = 0.72
print(f"Loaded results from {json_path}")

# ===================== Scatter Output Dir ==================
save_dir = f"plots_scatter_{PREFIX}_{THRESHOLD}"
os.makedirs(save_dir, exist_ok=True)

# ====================== Configurations ======================
L_values = ["L8", "L16", "L24"]
lam_values = [
    "BASELINE",
    "lam-2p0", "lam-1p5", "lam-1p0", "lam-0p5",
    "lam0p5", "lam1p0", "lam1p5", "lam2p0"
]

from scipy.stats import pearsonr

def compute_corr(x_list, y_list):
    """
    x_list: list of metric values (length N)
    y_list: list of final accuracy labels (0 or 1) (length N)
    return Pearson correlation, or None
    """
    x = np.array(x_list, dtype=float)
    y = np.array(y_list, dtype=float)

    if len(x) < 3 or np.std(x) == 0:
        return None  # correlation undefined

    try:
        return float(pearsonr(x, y)[0])
    except:
        return None

metric_names = [
    # "corr_full",
    "corr_hard",
    "corr_avg_prefix",
    "corr_avg_steps",
    "corr_avg_first_error",
    "avg_prefix",
    "avg_first_error",
    "avg_steps"
]

metric_display = {
    # "corr_full": "Pearson_F_full",
    "corr_hard": "Pearson_F_hard",
    "corr_avg_prefix": "Prefix_Correlation",
    "corr_avg_steps": "Length_Correlation",
    "corr_avg_first_error": "First_Error_Correlation",
    "avg_prefix": "Average Prefix Length",
    "avg_first_error": "Average First Error",
    "avg_steps": "Average #Steps",
    "avg_scores_and_final_acc": "Average Scores vs Final Accuracy",
    "avg_tokens_per_step_and_final_acc": "Average Tokens per Step vs Final Accuracy"
}

# Markers for λ
lam_markers = {
    "BASELINE": "o",
    # negative λ
    "lam-0p5": "P",
    "lam-1p0": "D",
    "lam-1p5": "<",
    "lam-2p0": ">",
    # positive λ
    "lam0p5": "s",
    "lam1p0": "^",
    "lam1p5": "v",
    "lam2p0": "X"
}

# Colors for layers
L_colors = {
    "L8": "tab:blue",
    "L16": "tab:orange",
    "L24": "tab:green",
}

# ====================== SUPPORT MULTIPLE MODELS ======================
model_names = list(results.keys())
print("Detected model names:", model_names)

# ====================== λ parser ==========================
def lam_to_float(lam: str) -> float:
    if lam == "BASELINE":
        return 0.0
    core = lam[3:].replace("p", ".")
    return float(core)




# ============================================================
# Unified function to recompute prefix and first-error using thr=0.7
# ============================================================

def recompute_prefix_and_first_error(step_scores_list, thr=0.7):
    """
    step_scores_list: list of list of scores for samples
                      e.g., [[0.9,0.8,0.4], [0.95,0.7,0.72,0.1], ...]

    Returns:
        prefix_list: list of prefix lengths per sample
        first_error_list: list of first error step (1-indexed), or None
    """

    prefix_list = []
    first_error_list = []

    for scores in step_scores_list:
        prefix_len = 0
        first_error = None

        for i, s in enumerate(scores):
            if s >= thr:                   # ★ 统一为 >=
                prefix_len += 1
            else:
                first_error = i + 1       # ★ 1-indexed
                break

        # 如果完全没有错误
        if first_error is None:
            # 你可以选择 first_error = None 或 len(scores)+1
            first_error = len(scores) + 1 # first_error = None   first_error = len(scores) + 1

        prefix_list.append(prefix_len)
        first_error_list.append(first_error)

    return prefix_list, first_error_list

corr_results = {}  # save new corr here if you want
# ====================== Draw Scatter for Metrics ======================
for model_name in model_names:

    print(f"\n====== Plotting for model: {model_name} ======")

    for metric in metric_names:
        plt.figure(figsize=(8,6))

        for L in L_values:
            if L not in results[model_name]:
                continue

            for lam in lam_values:
                if lam not in results[model_name][L]:
                    continue

                entry = results[model_name][L][lam]

                # 如果没有 step_scores，则跳过
                if "step_scores" not in entry:
                    continue

                # ★★★ 重新计算 prefix 和 first error，threshold=0.7 ★★★
                prefix_list, first_error_list = recompute_prefix_and_first_error(
                    entry["step_scores"], thr=THRESHOLD
                )
                avg_prefix_recomputed = np.mean(prefix_list)

                # 处理 None
                first_error_clean = [
                    fe if fe is not None else (
                        max([x for x in first_error_list if x is not None]) + 1
                    )
                    for fe in first_error_list
                ]
                avg_first_error_recomputed = np.mean(first_error_clean)

                step_scores = entry["step_scores"]      # list[list[float]]
                step_texts  = entry["step_texts"]       # list[list[str]]
                Y = np.array(entry["Y"], dtype=float)   # 0/1 labels

                # ---------- Recompute metrics per sample ----------
                prefix_list, first_error_list = recompute_prefix_and_first_error(step_scores, thr=THRESHOLD)
                
                # clean first error
                first_error_clean = [
                    fe if fe is not None else (max([x for x in first_error_list if x is not None]) + 1)
                    for fe in first_error_list
                ]

                avg_scores_per_sample = [np.mean(s) for s in step_scores]
                avg_steps_list = [len(s) for s in step_scores]
                avg_prefix_list = prefix_list
                avg_first_error_list = first_error_clean

                # hard correctness (step1 >= THRESHOLD)
                F_hard = [1 if s[0] >= THRESHOLD else 0 for s in step_scores]

                # ---------- Compute correlations ----------
                corr_hard = compute_corr(F_hard, Y)
                corr_avg_prefix = compute_corr(avg_prefix_list, Y)
                corr_avg_steps = compute_corr(avg_steps_list, Y)
                corr_avg_first_error = compute_corr(avg_first_error_list, Y)

                # Save the updated corr if needed
                entry["corr_hard_new"] = corr_hard
                entry["corr_avg_prefix_new"] = corr_avg_prefix
                entry["corr_avg_steps_new"] = corr_avg_steps
                entry["corr_avg_first_error_new"] = corr_avg_first_error

                # print update to screen
                print(f"[{model_name}] {L}-{lam}: corr_prefix={corr_avg_prefix:.4f}, corr_steps={corr_avg_steps:.4f}")
                # ====================================================
                # ★ 根据 metric 决定 x 的值 ★
                # ====================================================
                if metric == "avg_prefix":
                    x = avg_prefix_recomputed
                elif metric == "avg_first_error":
                    x = avg_first_error_recomputed
                else:
                    # 其他 metric 仍然使用原值
                    if metric == "corr_avg_prefix":
                        x = entry["corr_avg_prefix_new"]
                    elif metric == "corr_avg_steps":
                        x = entry["corr_avg_steps_new"]
                    elif metric == "corr_avg_first_error":
                        x = entry["corr_avg_first_error_new"]
                    elif metric == "corr_hard":
                        x = entry["corr_hard_new"]
                    else:
                        x = entry[metric]


                # y 是 final accuracy
                y = np.mean(entry["Y"])

                plt.scatter(
                    x, y,
                    color=L_colors[L],
                    marker=lam_markers.get(lam, "o"),
                    s=90,
                    label=f"{L}-{lam}"
                )

        plt.xlabel(metric_display.get(metric, metric))
        plt.ylabel("Final Accuracy")
        plt.title(f"[{model_name}] {metric_display.get(metric, metric)} vs Accuracy")

        handles, labels = plt.gca().get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        plt.legend(uniq.values(), uniq.keys(), bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.grid(alpha=0.3)
        plt.tight_layout()

        out_path = os.path.join(save_dir, f"{model_name}_scatter_{metric}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"📁 Saved: {out_path}")


    # =================================================================
    # 2. 新增：First-Step PRM correctness vs λ
    # =================================================================
    print(f"→ Plotting first-step correctness vs λ for {model_name}")

    plt.figure(figsize=(8,6))

    for L in L_values:
        if L not in results[model_name]:
            continue

        lam_x = []
        first_soft_y = []

        for lam in lam_values:
            if lam not in results[model_name][L]:
                continue

            entry = results[model_name][L][lam]

            if "step_scores" not in entry:
                continue

            step_scores = entry["step_scores"]

            # 取每个样本的 first step score
            fscores = [s[0] for s in step_scores if len(s) > 0]
            if len(fscores) == 0:
                continue

            lam_x.append(lam_to_float(lam))
            first_soft_y.append(float(np.mean(fscores)))

        if len(lam_x) == 0:
            continue

        order = np.argsort(lam_x)
        lam_x = np.array(lam_x)[order]
        first_soft_y = np.array(first_soft_y)[order]

        plt.scatter(
            lam_x,
            first_soft_y,
            s=90,
            color=L_colors[L],
            marker="o",
            label=f"{L}"
        )

    plt.xlabel("λ (steering strength)")
    plt.ylabel("Avg First-Step PRM Score")
    plt.title(f"[{model_name}] First-Step Correctness vs λ")
    plt.grid(alpha=0.3)
    plt.legend(title="Layer", bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"{model_name}_first_step_vs_lambda.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📁 Saved first-step plot: {out_path}")
    # =================================================================
    # 3. 新增：Avg Step Score per (L, λ) vs Final Accuracy
    # =================================================================
    print(f"→ Plotting avg step score vs final accuracy for {model_name}")

    plt.figure(figsize=(8,6))

    for L in L_values:
        if L not in results[model_name]:
            continue
        
        for lam in lam_values:
            if lam not in results[model_name][L]:
                continue

            entry = results[model_name][L][lam]

            if "step_scores" not in entry:
                continue

            # 所有样本的平均 step score
            avg_scores_per_sample = [np.mean(s) for s in entry["step_scores"]]
            if len(avg_scores_per_sample) == 0:
                continue

            avg_score = float(np.mean(avg_scores_per_sample))  # ★ 对所有 sample 求平均
            final_acc = float(np.mean(entry["Y"]))             # ★ 对所有 sample 求平均 accuracy

            plt.scatter(
                avg_score,
                final_acc,
                s=120,
                color=L_colors[L],
                marker=lam_markers.get(lam, "o"),
                alpha=0.8,
                label=f"{L}-{lam}"
            )

    plt.xlabel("Average Step-Correctness Score (mean over samples)")
    plt.ylabel("Final Accuracy")
    plt.title(f"[{model_name}] Avg Step Score vs Final Accuracy")
    plt.grid(alpha=0.3)

    # 去重 legend
    handles, labels = plt.gca().get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    plt.legend(uniq.values(), uniq.keys(), bbox_to_anchor=(1.05,1), loc="upper left")

    plt.tight_layout()

    out_path = os.path.join(save_dir, f"{model_name}_avg_step_score_vs_accuracy.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📁 Saved avg_step_score_vs_accuracy plot: {out_path}")


    # =================================================================
    # 4. 新增：Avg Tokens per Step per (L, λ) vs Final Accuracy
    # =================================================================
    print(f"→ Plotting avg tokens per step vs final accuracy for {model_name}")

    plt.figure(figsize=(8,6))

    for L in L_values:
        if L not in results[model_name]:
            continue
        
        for lam in lam_values:
            if lam not in results[model_name][L]:
                continue

            entry = results[model_name][L][lam]

            if "step_texts" not in entry:
                continue

            step_texts = entry["step_texts"]

            # 每个 sample 的每步 token 数
            avg_tokens_per_sample = []
            for steps in step_texts:
                token_counts = [len(step.split()) for step in steps]
                avg_tokens_per_sample.append(np.mean(token_counts))

            if len(avg_tokens_per_sample) == 0:
                continue

            avg_tokens = float(np.mean(avg_tokens_per_sample))  # ★（L,λ）的 token 平均
            final_acc = float(np.mean(entry["Y"]))              # ★（L,λ）的 accuracy 平均

            plt.scatter(
                avg_tokens,
                final_acc,
                s=120,
                color=L_colors[L],
                marker=lam_markers.get(lam, "o"),
                alpha=0.8,
                label=f"{L}-{lam}"
            )

    plt.xlabel("Average Tokens per Step (mean over samples)")
    plt.ylabel("Final Accuracy")
    plt.title(f"[{model_name}] Avg Tokens per Step vs Final Accuracy")
    plt.grid(alpha=0.3)

    handles, labels = plt.gca().get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    plt.legend(uniq.values(), uniq.keys(), bbox_to_anchor=(1.05,1), loc="upper left")

    plt.tight_layout()

    out_path = os.path.join(save_dir, f"{model_name}_avg_tokens_vs_accuracy.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📁 Saved avg_tokens_vs_accuracy plot: {out_path}")

print("\n🎉 All model scatter plots saved!")
