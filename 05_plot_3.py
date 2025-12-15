import json
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# Load merged results
# ============================================================
PREFIX = "qwen_family"
json_path = f"/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/prm_out_{PREFIX}/results_merged.json"
#"/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/prm_out/results_merged.json"

with open(json_path, "r") as f:
    results = json.load(f)

print(f"Loaded: {json_path}")

# Output folder
save_dir = f"plots_correct_vs_wrong_{PREFIX}"
os.makedirs(save_dir, exist_ok=True)

# ============================================================
# config
# ============================================================
L_values = ["L8", "L16", "L24"]
lam_values = [
    "BASELINE",
    "lam-2p0", "lam-1p5", "lam-1p0", "lam-0p5",
    "lam0p5", "lam1p0", "lam1p5", "lam2p0"
]

L_colors = {
    "L8": "tab:blue",
    "L16": "tab:orange",
    "L24": "tab:green",
}

# ============================================================
# helper functions
# ============================================================
def lam_to_float(lam):
    if lam == "BASELINE":
        return 0.0
    core = lam[3:].replace("p", ".")
    return float(core)

def recompute_prefix_and_first_error(step_scores_list, thr=0.7):
    prefix_list = []
    first_error_list = []
    for scores in step_scores_list:
        prefix_len = 0
        first_error = None
        for i, s in enumerate(scores):
            if s >= thr:
                prefix_len += 1
            else:
                first_error = i + 1
                break
        if first_error is None:
            first_error = None
        prefix_list.append(prefix_len)
        first_error_list.append(first_error)
    return prefix_list, first_error_list


# ============================================================
# Start plotting per model
# ============================================================
for model_name in results.keys():
    print(f"\n=== Model: {model_name} ===")

    # --------------------------------------------------------
    # Define metrics you want to compare correct vs wrong
    # --------------------------------------------------------
    metric_names = {
        "prefix": "Avg Prefix Length (thr=0.7)",
        "first_error": "Avg First Error Step (thr=0.7)",
        "avg_score": "Avg Step Correctness",
        "avg_tokens": "#Tokens per Step",
        "avg_steps": "Avg Total Steps"
    }

    for metric_key, metric_title in metric_names.items():

        plt.figure(figsize=(10,6))

        # =========================
        # For each layer
        # =========================
        for L in L_values:
            if L not in results[model_name]:
                continue

            lam_nums = []
            metric_correct = []
            metric_wrong = []

            for lam in lam_values:
                if lam not in results[model_name][L]:
                    continue

                entry = results[model_name][L][lam]
                step_scores = entry["step_scores"]
                step_texts = entry["step_texts"]
                Y = entry["Y"]

                # compute metrics
                prefix_list, first_error_list = recompute_prefix_and_first_error(step_scores)

                # 清洗 first error
                fe_clean = [
                    fe if fe is not None else (max([x for x in first_error_list if x is not None]) + 1)
                    for fe in first_error_list
                ]

                # avg correctness
                avg_scores_per_sample = [np.mean(s) for s in step_scores]

                # avg tokens
                avg_tokens_per_sample = []
                for steps in step_texts:
                    token_counts = [len(step.split()) for step in steps]
                    avg_tokens_per_sample.append(np.mean(token_counts))

                # group by Y
                Y = np.array(Y)

                if metric_key == "prefix":
                    vals_correct = np.mean(np.array(prefix_list)[Y==1]) if np.any(Y==1) else None
                    vals_wrong   = np.mean(np.array(prefix_list)[Y==0]) if np.any(Y==0) else None

                elif metric_key == "first_error":
                    vals_correct = np.mean(np.array(fe_clean)[Y==1]) if np.any(Y==1) else None
                    vals_wrong   = np.mean(np.array(fe_clean)[Y==0]) if np.any(Y==0) else None

                elif metric_key == "avg_score":
                    vals_correct = np.mean(np.array(avg_scores_per_sample)[Y==1]) if np.any(Y==1) else None
                    vals_wrong   = np.mean(np.array(avg_scores_per_sample)[Y==0]) if np.any(Y==0) else None

                elif metric_key == "avg_tokens":
                    vals_correct = np.mean(np.array(avg_tokens_per_sample)[Y==1]) if np.any(Y==1) else None
                    vals_wrong   = np.mean(np.array(avg_tokens_per_sample)[Y==0]) if np.any(Y==0) else None
                
                elif metric_key == "avg_steps":
                    total_steps_per_sample = np.array([len(s) for s in step_scores])
                    vals_correct = np.mean(total_steps_per_sample[Y==1]) if np.any(Y==1) else None
                    vals_wrong   = np.mean(total_steps_per_sample[Y==0]) if np.any(Y==0) else None

                lam_nums.append(lam_to_float(lam))
                metric_correct.append(vals_correct)
                metric_wrong.append(vals_wrong)

            # 排序
            lam_nums = np.array(lam_nums)
            idx = np.argsort(lam_nums)
            lam_nums = lam_nums[idx]
            metric_correct = np.array(metric_correct)[idx]
            metric_wrong = np.array(metric_wrong)[idx]

            # 绘图
            plt.plot(lam_nums, metric_correct, "-o", color=L_colors[L], label=f"{L} (Correct)")
            plt.plot(lam_nums, metric_wrong, "-x", color=L_colors[L], linestyle="--", label=f"{L} (Wrong)")

        plt.xlabel("λ value")
        plt.ylabel(metric_title)
        plt.title(f"{model_name} — {metric_title}\nCorrect vs Wrong Comparison")
        plt.grid(alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")

        out_path = os.path.join(save_dir, f"{model_name}_{metric_key}_correct_vs_wrong.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"📁 Saved: {out_path}")

print("\n🎉 All correct-vs-wrong comparison plots saved!")
