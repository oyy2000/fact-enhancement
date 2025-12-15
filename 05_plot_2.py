# plot_multi_model.py
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# ===========================================================
#                Load JSON Results
# ===========================================================
# INPUT_JSON = "results_merged.json"   # 修改成你的文件名
INPUT_JSON = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/prm_out_qwen_family/results_merged.json"
with open(INPUT_JSON, "r") as f:
    results = json.load(f)

print("Loaded models:", list(results.keys()))

# ===========================================================
#                Detect All Dimensions
# ===========================================================

# detect models
model_names = list(results.keys())

# detect L (layers)
L_values = sorted(
    {L for model in results.values() for L in model.keys()},
    key=lambda x: int(x.replace("L",""))
)

# detect lam keys
lam_values = sorted(
    {lam for model in results.values() for L in model.values() for lam in L.keys()},
    key=lambda x: x
)

print("Detected Layers:", L_values)
print("Detected Lambdas:", lam_values)

# ===========================================================
#            Config: Human-readable λ labels
# ===========================================================
lam_plot_labels = {
    "lam-2p0":"-2.0",
    "lam-1p5":"-1.5",
    "lam-1p0":"-1.0",
    "lam-0p5":"-0.5",
    "BASELINE":"0.0",
    "lam0p5":"0.5",
    "lam1p0":"1.0",
    "lam1p5":"1.5",
    "lam2p0":"2.0"
}

# x-axis ticks
x = [lam_plot_labels.get(lam, lam) for lam in lam_values]

# ===========================================================
#             Output directory for figures
# ===========================================================
fig_dir = "figures"
os.makedirs(fig_dir, exist_ok=True)
print(f"📁 Figures will be saved under {fig_dir}/")

# ===========================================================
#                 Metrics to Plot
# ===========================================================
metric_names = [
    ("corr_full",            "Correlation (Full Step Mean)",                "Pearson r", "corr_full"),
    ("corr_hard",            "Correlation (Hard Step Score)",               "Pearson r", "corr_hard"),
    ("corr_avg_prefix",      "Correlation (Avg Prefix Correctness Length)", "Pearson r", "corr_avg_prefix"),
    ("corr_avg_first_error", "Correlation (Avg First Error Position)",      "Pearson r", "corr_avg_first_error"),
    ("corr_avg_steps",       "Correlation (Mean Total Steps)",              "Pearson r", "corr_avg_steps"),
    ("avg_prefix",           "Average Prefix Correctness Length",           "Prefix Length", "avg_prefix"),
    ("avg_first_error",      "Average First Error Position",                "First Error Step", "avg_first_error"),
    ("avg_steps",            "Average Total Steps",                         "Step Count", "avg_steps"),
]

# ===========================================================
#                  Plotting Function
# ===========================================================
def plot_metric(metric_key, title, ylabel, fname):

    plt.figure(figsize=(8,6))

    for model_name in model_names:
        for L in L_values:
            if L not in results[model_name]:
                continue

            y = [
                results[model_name][L][lam][metric_key]
                if (lam in results[model_name][L]
                    and metric_key in results[model_name][L][lam])
                else None
                for lam in lam_values
            ]

            plt.plot(
                x, y,
                marker="o",
                label=f"{model_name} | {L}"
            )

    plt.title(title, fontsize=14)
    plt.xlabel("Lambda", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(alpha=0.3)

    # unique legend
    handles, labels = plt.gca().get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    plt.legend(uniq.values(), uniq.keys(), title="Model | Layer", fontsize=9, bbox_to_anchor=(1.05,1), loc="upper left")

    save_path = f"{fig_dir}/{fname}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📌 Saved: {save_path}")


# ===========================================================
#                Generate All Figures
# ===========================================================
for metric_key, title, ylabel, fname in metric_names:
    plot_metric(metric_key, title, ylabel, fname)

print("\n🎉 All plots completed & saved!")
