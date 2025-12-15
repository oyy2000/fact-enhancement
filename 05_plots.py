import json
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr


# ============================================================
# CONFIG
# ============================================================

THRESHOLD = 0.72

MODEL_FILES = [
    "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/prm_out_qwen_family/results_merged.json",
    # 你可以继续添加其它模型族
    # "/path/to/prm_out_llama/results_merged.json",
]

SAVE_ROOT = "plots_final_v2"
os.makedirs(SAVE_ROOT, exist_ok=True)
os.makedirs(f"{SAVE_ROOT}/correct_wrong", exist_ok=True)
os.makedirs(f"{SAVE_ROOT}/scatter", exist_ok=True)
os.makedirs(f"{SAVE_ROOT}/special", exist_ok=True)

MODEL_MARKERS = ["o", "s", "^", "D", "P", "X", "<", ">", "v"]
WRONG_MARKERS = ["x", "X", "v", "p", "*", "d", "h", "H", "+"]

MODEL_LINESTYLES = ["-", "--", "-.", ":"]
LAYER_ALPHA = [1.0, 0.75, 0.55, 0.35]


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


def extract_metrics(entry, thr):
    step_scores = entry["step_scores"]
    step_texts = entry["step_texts"]
    Y = np.array(entry["Y"])

    prefix, first_err = compute_prefix_first_error(step_scores, thr)

    avg_scores = np.array([np.mean(s) for s in step_scores])
    avg_steps = np.array([len(s) for s in step_scores])
    avg_tokens = np.array([
        np.mean([len(t.split()) for t in steps]) for steps in step_texts
    ])
    
    F_hard = np.array([1 if s[0] >= thr else 0 for s in step_scores])

    return {
        "prefix": prefix,
        "first_error": first_err,
        "avg_scores": avg_scores,
        "avg_steps": avg_steps,
        "avg_tokens": avg_tokens,
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
        "avg_tokens": "Avg Tokens per Step",
        "avg_steps": "Avg Total Steps"
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
                    M = extract_metrics(entry, THRESHOLD)
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
#  B. Scatter plots
# ============================================================

def plot_scatter():

    metrics = {
        "avg_prefix": "Avg Prefix Length",
        "avg_first_error": "Avg First Error Step",
        "avg_steps": "Avg Total Steps",
        "corr_avg_prefix": "Prefix Correlation",
        "corr_avg_steps": "Steps Correlation",
        "corr_avg_first_error": "First Error Correlation",
        "corr_hard": "Hard First Step Correlation"
    }

    outdir = f"{SAVE_ROOT}/scatter"

    for metric, metric_name in metrics.items():

        plt.figure(figsize=(10,6))

        for model_name, model_data in model_results.items():
            layers = detect_layers(model_data)
            alpha_map = {L: LAYER_ALPHA[i % len(LAYER_ALPHA)] for i, L in enumerate(layers)}

            for L in layers:
                lambdas = detect_lambdas(model_data, L)

                for lam in lambdas:
                    entry = model_data[L][lam]
                    M = extract_metrics(entry, THRESHOLD)
                    Y = M["Y"]
                    acc = np.mean(Y)

                    prefix = M["prefix"]
                    steps = M["avg_steps"]
                    first = M["first_error"]
                    hard = M["F_hard"]

                    corr_prefix = compute_corr(prefix, Y)
                    corr_steps  = compute_corr(steps, Y)
                    corr_fe     = compute_corr(first, Y)
                    corr_h      = compute_corr(hard, Y)

                    # choose X based on metric
                    if metric == "avg_prefix":
                        x = np.mean(prefix)
                    elif metric == "avg_first_error":
                        x = np.mean(first)
                    elif metric == "avg_steps":
                        x = np.mean(steps)
                    elif metric == "corr_avg_prefix":
                        x = corr_prefix
                    elif metric == "corr_avg_steps":
                        x = corr_steps
                    elif metric == "corr_avg_first_error":
                        x = corr_fe
                    elif metric == "corr_hard":
                        x = corr_h

                    plt.scatter(
                        x, acc,
                        color=MODEL_COLOR_MAP[model_name],
                        marker=MODEL_MARKER_MAP[model_name],
                        s=140,
                        alpha=alpha_map[L],
                        label=f"{model_name}-{L}"
                    )

        plt.xlabel(metric_name)
        plt.ylabel("Accuracy")
        plt.title(metric_name + " vs Accuracy")
        plt.grid(alpha=0.3)

        handles, labels = plt.gca().get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        plt.legend(uniq.values(), uniq.keys(), bbox_to_anchor=(1.05,1), loc="upper left")

        plt.tight_layout()
        out_path = f"{outdir}/{metric}.png"
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

    for model_name, model_data in model_results.items():
        layers = detect_layers(model_data)

        for L in layers:
            for lam, entry in model_data[L].items():
                avg_step_score = np.mean([np.mean(s) for s in entry["step_scores"]])
                acc = np.mean(entry["Y"])

                plt.scatter(
                    avg_step_score,
                    acc,
                    color=MODEL_COLOR_MAP[model_name],
                    marker=MODEL_MARKER_MAP[model_name],
                    s=140,
                    alpha=0.85
                )

    plt.xlabel("Avg Step Score")
    plt.ylabel("Accuracy")
    plt.title("Avg Step Score vs Accuracy")
    plt.grid(alpha=0.3)

    out_path = f"{SAVE_ROOT}/special/avg_step_score_vs_accuracy.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)


def plot_avg_tokens_vs_acc():
    plt.figure(figsize=(10,6))

    for model_name, model_data in model_results.items():
        layers = detect_layers(model_data)

        for L in layers:
            for lam, entry in model_data[L].items():
                avg_tokens = np.mean([
                    np.mean([len(t.split()) for t in steps])
                    for steps in entry["step_texts"]
                ])
                acc = np.mean(entry["Y"])

                plt.scatter(
                    avg_tokens,
                    acc,
                    color=MODEL_COLOR_MAP[model_name],
                    marker=MODEL_MARKER_MAP[model_name],
                    s=140,
                    alpha=0.85
                )

    plt.xlabel("Avg Tokens per Step")
    plt.ylabel("Accuracy")
    plt.title("Avg Tokens vs Accuracy")
    plt.grid(alpha=0.3)

    out_path = f"{SAVE_ROOT}/special/avg_tokens_vs_accuracy.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)


# ============================================================
# Run All
# ============================================================

print("\n=== Generating Correct-vs-Wrong plots ===")
plot_correct_wrong()

print("\n=== Generating Scatter plots ===")
plot_scatter()

print("\n=== Generating Special plots ===")
plot_first_step_vs_lambda()
plot_avg_score_vs_acc()
plot_avg_tokens_vs_acc()

print("\n🎉 FINAL VERSION COMPLETE!  Saved in:", SAVE_ROOT)
