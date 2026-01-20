import json
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
from transformers import AutoTokenizer
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CONFIG
# ============================================================

LAM_MARKERS = ["o", "s", "^", "D", "P", "X", "<", ">", "v"]

def get_lambda_marker_map(lambdas):
    lambdas = sorted(lambdas, key=lam_to_float)
    return {lam: LAM_MARKERS[i % len(LAM_MARKERS)] for i, lam in enumerate(lambdas)}

THRESHOLD = 0.9
def get_layer_color_map(layers):
    colors = plt.cm.tab10(np.linspace(0, 1, len(layers)))
    return {L: colors[i] for i, L in enumerate(layers)}

SAVE_DIR = "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_less_tokens_3B_lib_manual_same_lib_1000samples/prm_results"
MODEL_FILES = [
    f"{SAVE_DIR}/results_merged.json",
    # 你可以继续添加其它模型族
    # "/path/to/prm_out_llama/results_merged.json",
]

SAVE_ROOT = f"{SAVE_DIR}"
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

    # ===== PPL / Rank Shift (Optional) =====
    ppl = np.array(entry.get("ppl", []))
    if len(ppl) == 0:
        ppl = np.full(len(Y), np.nan)
        
    rank_shift = np.array(entry.get("rank_shift", []))
    if len(rank_shift) == 0:
        rank_shift = np.full(len(Y), np.nan)

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
        "ppl": ppl,
        "rank_shift": rank_shift,
        "Y": Y
    }


# ============================================================
# Load All Models (each JSON may contain multiple models)
# ============================================================

def setup_plotting(data, save_dir):
    global model_results, all_models, MODEL_COLOR_MAP, MODEL_MARKER_MAP, MODEL_LINESTYLE_MAP, SAVE_ROOT
    model_results = data
    SAVE_ROOT = save_dir
    
    # Ensure output directories exist
    os.makedirs(SAVE_ROOT, exist_ok=True)
    os.makedirs(f"{SAVE_ROOT}/correct_wrong", exist_ok=True)
    os.makedirs(f"{SAVE_ROOT}/all", exist_ok=True)
    os.makedirs(f"{SAVE_ROOT}/scatter", exist_ok=True)
    os.makedirs(f"{SAVE_ROOT}/special", exist_ok=True)
    os.makedirs(f"{SAVE_ROOT}/per_step_avg", exist_ok=True)

    all_models = list(model_results.keys())
    print("\nTotal models:", len(all_models))

    # Assign Color, Marker, Linestyle Per Model
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

def main():
    print("Loading all models...")
    data = {}
    for file in MODEL_FILES:
        raw = json.load(open(file))
        for model_name, model_data in raw.items():
            print(model_name)
            data[model_name] = model_data
            print("Loaded model:", model_name)
    
    setup_plotting(data, SAVE_DIR)

    print("\n=== Generating Correct-vs-Wrong plots ===")
    plot_correct_wrong()

    print("\n=== Generating All-sample plots ===")
    plot_all()

    plot_avg_score_vs_acc()
    plot_avg_tokens_vs_acc()
    plot_avg_total_tokens_vs_acc()

    plot_per_step_avg_correctness(max_steps=15)

if __name__ == "__main__":
    main()

    metrics = {
        "prefix": "Avg Prefix Length",
        "first_error": "Avg First Error Step",
        "avg_scores": "Avg Step Correctness",
        "avg_steps": "Avg Total Steps",
        "avg_tokens_per_step": "Avg Tokens per Step",
        "avg_total_tokens": "Avg Total Tokens",
        "total_error_steps": "Total Error Steps",
        "error_step_ratio": "Error Step Ratio",
        "ppl": "Perplexity",
        "rank_shift": "Rank Shift"
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
        "error_step_ratio": "Error Step Ratio",
        "ppl": "Perplexity",
        "rank_shift": "Rank Shift"
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


def lam_to_size(lam, base=80, scale=40):
    return base + scale * abs(lam)


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
        "error_step_ratio": "Error Step Ratio",
        "ppl": "Perplexity",
        "rank_shift": "Rank Shift"
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
                    np.mean(steps)
                    for steps in entry["step_token_len"]
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

def _safe_name(s: str) -> str:
    """Make a string safe for filenames."""
    return (
        s.replace("/", "_")
         .replace(" ", "_")
         .replace(":", "_")
         .replace("__", "_")
    )

def plot_per_step_avg_correctness(max_steps=15):
    """
    For each (model, layer, lambda), plot mean step correctness vs step index.
    Fixed: Generate specific x-axis ranges for Correct/Wrong curves to avoid shape mismatch.
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

                # 计算均值
                mean_all, n_all = per_step_mean(step_scores_all, max_steps=max_steps)
                mean_corr, n_corr = per_step_mean(step_scores_corr, max_steps=max_steps)
                mean_wrong, n_wrong = per_step_mean(step_scores_wrong, max_steps=max_steps)

                if len(mean_all) == 0:
                    continue

                plt.figure(figsize=(9, 5))

                # 1. Plot ALL (为 All 生成专属 steps)
                steps_all = np.arange(1, len(mean_all) + 1)
                plt.plot(steps_all, mean_all, marker="o", linewidth=2, label=f"All (n@1={n_all[0]})")

                # 2. Plot CORRECT (为 Correct 生成专属 steps)
                if len(mean_corr) > 0:
                    steps_corr = np.arange(1, len(mean_corr) + 1)
                    plt.plot(steps_corr, mean_corr, marker="s", linewidth=2, label=f"Correct (n@1={n_corr[0]})")

                # 3. Plot WRONG (为 Wrong 生成专属 steps - 修复点)
                if len(mean_wrong) > 0:
                    steps_wrong = np.arange(1, len(mean_wrong) + 1)  # <--- FIX: 独立生成 x 轴
                    plt.plot(steps_wrong, mean_wrong, marker="x", linewidth=2, label=f"Wrong (n@1={n_wrong[0]})")

                plt.axhline(THRESHOLD, linestyle="--", linewidth=1.2, label=f"thr={THRESHOLD}")
                plt.xlabel("Step k")
                plt.ylabel("Avg PRM Step Score (across samples)")
                plt.title(f"Per-step Avg Correctness\n{model_name} | L={L} | λ={lam}")
                plt.grid(alpha=0.25)
                plt.legend()

                fname = f"{_safe_name(model_name)}_L{L}_{_safe_name(lam)}_per_step_avg.png"
                out_path = os.path.join(outdir, fname)
                plt.tight_layout()
                plt.savefig(out_path, dpi=300)
                plt.close()
                print("Saved:", out_path)


# ============================================================
#  C.2 More Special plots (Tokens, Errors, Steps, Prefix)
# ============================================================

def plot_avg_total_tokens_vs_acc():
    """
    Plots the Average Total Tokens per Solution (sum of all steps) vs Accuracy.
    Useful to see if longer solutions correlate with lower accuracy (reasoning fatigue).
    """
    plt.figure(figsize=(10,6))
    seen_models = set()

    for model_name, model_data in model_results.items():
        layers = detect_layers(model_data)

        for L in layers:
            lambdas = detect_lambdas(model_data, L)
            for lam in lambdas:
                entry = model_data[L][lam]
                
                # Calculate total tokens per sample (sum of tokens in all steps)
                # Using split() as a proxy for tokens to be consistent with your previous code
                total_tokens_per_sample = [
                    np.sum(steps) for steps in entry["step_token_len"]
                ]
                
                avg_total_tokens = np.mean(total_tokens_per_sample)
                acc = np.mean(entry["Y"])
                
                label = model_name if model_name not in seen_models else None
                seen_models.add(model_name)

                plt.scatter(
                    avg_total_tokens,
                    acc,
                    color=MODEL_COLOR_MAP[model_name],
                    marker=MODEL_MARKER_MAP[model_name],
                    s=140,
                    alpha=0.85,
                    label=label
                )

    plt.xlabel("Avg Total Tokens per Solution")
    plt.ylabel("Accuracy")
    plt.title("Avg Total Tokens (Solution Length) vs Accuracy")
    plt.grid(alpha=0.3)
    plt.legend(title="Model")

    out_path = f"{SAVE_ROOT}/special/avg_total_tokens_vs_accuracy.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)


def plot_avg_steps_vs_acc():
    """
    Plots Average Number of Steps (Depth) vs Accuracy.
    """
    plt.figure(figsize=(10,6))
    seen_models = set()

    for model_name, model_data in model_results.items():
        layers = detect_layers(model_data)

        for L in layers:
            lambdas = detect_lambdas(model_data, L)
            for lam in lambdas:
                entry = model_data[L][lam]
                
                # Calculate number of steps per sample
                num_steps_per_sample = [len(s) for s in entry["step_token_len"]]
                
                avg_steps = np.mean(num_steps_per_sample)
                acc = np.mean(entry["Y"])
                
                label = model_name if model_name not in seen_models else None
                seen_models.add(model_name)

                plt.scatter(
                    avg_steps,
                    acc,
                    color=MODEL_COLOR_MAP[model_name],
                    marker=MODEL_MARKER_MAP[model_name],
                    s=140,
                    alpha=0.85,
                    label=label
                )

    plt.xlabel("Avg Number of Steps")
    plt.ylabel("Accuracy")
    plt.title("Avg Step Count (Depth) vs Accuracy")
    plt.grid(alpha=0.3)
    plt.legend(title="Model")

    out_path = f"{SAVE_ROOT}/special/avg_num_steps_vs_accuracy.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)


def plot_avg_error_steps_vs_acc():
    """
    Plots Average Number of 'Error Steps' (Score < THRESHOLD) per solution vs Accuracy.
    """
    plt.figure(figsize=(10,6))
    seen_models = set()

    for model_name, model_data in model_results.items():
        layers = detect_layers(model_data)

        for L in layers:
            lambdas = detect_lambdas(model_data, L)
            for lam in lambdas:
                entry = model_data[L][lam]
                
                # Count steps below threshold per sample
                error_counts = [
                    sum(1 for score in steps if score < THRESHOLD)
                    for steps in entry["step_scores"]
                ]
                
                avg_errors = np.mean(error_counts)
                acc = np.mean(entry["Y"])
                
                label = model_name if model_name not in seen_models else None
                seen_models.add(model_name)

                plt.scatter(
                    avg_errors,
                    acc,
                    color=MODEL_COLOR_MAP[model_name],
                    marker=MODEL_MARKER_MAP[model_name],
                    s=140,
                    alpha=0.85,
                    label=label
                )

    plt.xlabel(f"Avg Error Steps (Score < {THRESHOLD})")
    plt.ylabel("Accuracy")
    plt.title(f"Avg Error Steps per Solution vs Accuracy")
    plt.grid(alpha=0.3)
    plt.legend(title="Model")

    out_path = f"{SAVE_ROOT}/special/avg_error_steps_vs_accuracy.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)
 