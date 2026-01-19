import argparse
import os
import json
import glob
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib.lines import Line2D

# ============================================================
# 1. GLOBAL CONFIGURATION & PATHS
# ============================================================

# User Paths
PREFIX = "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_less_tokens_3B_lib_manual_same_lib"
FOLDER = os.path.join(PREFIX, "prm")
BASE_DIR = os.path.join(PREFIX, "gsm8k_cot_zeroshot") #"gsm8k_cot_zeroshot"
PRM_MODEL = "Qwen/Qwen2.5-Math-PRM-7B"

# Ensure directories exist
os.makedirs(FOLDER, exist_ok=True)
SAVE_ROOT = FOLDER  # Plotting will save here
for sub in ["correct_wrong", "scatter", "special", "all", "per_step_avg"]:
    os.makedirs(os.path.join(SAVE_ROOT, sub), exist_ok=True)

# GPU Settings
NUM_GPUS = 8
GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]

# Lambda Generation
STEP = 0.5
STEER_LAMBDAS = [i * STEP for i in range(-5, 6)]  # -5 to 5
STEER_LAMBDAS = [round(x, 2) for x in STEER_LAMBDAS]
STEER_LAMBDAS = [0.0]

def lam_to_str(lam: float) -> str:
    if abs(lam) < 1e-9:
        return "BASELINE"
    sign = "-" if lam < 0 else ""
    s = f"{abs(lam):.2f}"          # 1.0 -> "1.00"
    s = s.rstrip("0")              # "1.00" -> "1."
    if s.endswith("."):
        s += "0"                   # "1." -> "1.0"
    s = s.replace(".", "p")        # "1.0" -> "1p0"
    return f"lam{sign}{s}"

lam_values = [lam_to_str(lam) for lam in STEER_LAMBDAS]
print("Lambda values:", lam_values)

# Model Definitions
MODEL_MAP = {
    "Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
}

MODEL_TO_LAYERS = {
    "Qwen2.5-3B-Instruct": [1], #[18, 32, 36],
}

MODEL_FOLDER_MAP = {k: f"Qwen__{k}" for k in MODEL_MAP}

# ============================================================
# 2. PLOTTING CONFIGURATION & HELPERS
# ============================================================

THRESHOLD = 0.9

# Styling
MODEL_MARKERS = ["o", "s", "^", "D", "P", "X", "<", ">", "v"]
WRONG_MARKERS = ["x", "X", "v", "p", "*", "d", "h", "H", "+"]
MODEL_LINESTYLES = ["-", "--", "-.", ":"]
LAYER_ALPHA = [1.0, 0.7, 0.5, 0.2]

def lam_to_float(lam):
    if lam == "BASELINE":
        return 0.0
    return float(lam[3:].replace("p", "."))

def detect_layers(model_data):
    return list(model_data.keys())

def detect_lambdas(model_data, L):
    return list(model_data[L].keys())

def _safe_name(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_").replace(":", "_").replace("__", "_")

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

def extract_metrics(entry, thr, model_name=None):
    step_scores = entry["step_scores"]
    step_token_len = entry["step_token_len"]
    Y = np.array(entry["Y"])

    prefix, first_err = compute_prefix_first_error(step_scores, thr)

    avg_scores = np.array([np.mean(s) for s in step_scores])
    avg_steps = np.array([len(s) for s in step_scores])
    
    avg_tokens_per_step = np.array([np.mean(lens) for lens in step_token_len])
    avg_total_tokens = np.array([np.sum(lens) for lens in step_token_len])
    
    total_error_steps = np.array([np.sum(np.array(s) < thr) for s in step_scores])
    
    # Avoid div by zero
    error_step_ratio = np.array([
        np.sum(np.array(s) < thr) / max(len(s), 1) for s in step_scores
    ])
    
    F_hard = np.array([1 if s[0] >= thr else 0 for s in step_scores])

    return {
        "prefix": prefix,
        "first_error": first_err,
        "avg_scores": avg_scores,
        "avg_steps": avg_steps,
        "total_error_steps": total_error_steps,
        "error_step_ratio": error_step_ratio,
        "avg_tokens_per_step": avg_tokens_per_step,
        "avg_total_tokens": avg_total_tokens,
        "F_hard": F_hard,
        "Y": Y
    }

def per_step_mean(step_scores, max_steps=None):
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

# ============================================================
# 3. WORKER FUNCTION (EXECUTION)
# ============================================================

def worker(job_idx, job, gpu_id):
    model_name, gen_model_name, L, lam, jsonl = job
    out_file = f"{FOLDER}/results_chunk_{job_idx}.json"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = (
        f"python 07_run_prm_single.py "
        f"--model_name '{model_name}' "
        f"--gen_model '{gen_model_name}' "
        f"--layer {L} "
        f"--lam {lam} "
        f"--jsonl {jsonl} "
        f"--out {out_file} "
        f"--prm_model '{PRM_MODEL}'"
    )
    print(f"[GPU{gpu_id}] → {cmd}")
    os.system(cmd)

# ============================================================
# 4. PLOTTING FUNCTIONS
# ============================================================

def run_all_plots(model_results):
    """
    Main driver function to generate all plots based on the loaded model_results.
    """
    all_models = list(model_results.keys())
    print(f"Generating plots for models: {all_models}")

    # Setup Colors
    color_list = plt.cm.tab20(np.linspace(0, 1, len(all_models)))
    MODEL_COLOR_MAP = {model: color_list[i] for i, model in enumerate(all_models)}
    MODEL_MARKER_MAP = {model: MODEL_MARKERS[i % len(MODEL_MARKERS)] for i, model in enumerate(all_models)}
    MODEL_LINESTYLE_MAP = {model: MODEL_LINESTYLES[i % len(MODEL_LINESTYLES)] for i, model in enumerate(all_models)}

    # --- Plot A: Correct vs Wrong ---
    def plot_correct_wrong():
        metrics = {
            "prefix": "Avg Prefix Length",
            "first_error": "Avg First Error Step",
            "avg_scores": "Avg Step Correctness",
            "avg_steps": "Avg Total Steps",
            "avg_tokens_per_step": "Avg Tokens per Step",
            "total_error_steps": "Total Error Steps",
            "error_step_ratio": "Error Step Ratio"
        }
        outdir = f"{SAVE_ROOT}/correct_wrong"
        
        for metric_key, metric_name in metrics.items():
            plt.figure(figsize=(10,6))
            for i, (model_name, model_data) in enumerate(model_results.items()):
                wrong_marker = WRONG_MARKERS[i % len(WRONG_MARKERS)]
                layers = detect_layers(model_data)
                alpha_map = {L: LAYER_ALPHA[j % len(LAYER_ALPHA)] for j, L in enumerate(layers)}

                for L in layers:
                    lambdas = detect_lambdas(model_data, L)
                    lam_vals, corr_vals, wrong_vals = [], [], []

                    for lam in lambdas:
                        entry = model_data[L][lam]
                        M = extract_metrics(entry, THRESHOLD, model_name)
                        Y = M["Y"]
                        lam_vals.append(lam_to_float(lam))
                        corr_vals.append(np.mean(M[metric_key][Y == 1]))
                        wrong_vals.append(np.mean(M[metric_key][Y == 0]))

                    # Sort
                    idx = np.argsort(lam_vals)
                    lam_vals = np.array(lam_vals)[idx]
                    corr_vals = np.array(corr_vals)[idx]
                    wrong_vals = np.array(wrong_vals)[idx]

                    plt.plot(lam_vals, corr_vals, color=MODEL_COLOR_MAP[model_name], 
                             linestyle=MODEL_LINESTYLE_MAP[model_name], marker=MODEL_MARKER_MAP[model_name],
                             markersize=7, alpha=alpha_map[L], label=f"{model_name}-{L} (Correct)")
                    
                    plt.plot(lam_vals, wrong_vals, color=MODEL_COLOR_MAP[model_name],
                             linestyle=MODEL_LINESTYLE_MAP[model_name], marker=wrong_marker,
                             markersize=8, alpha=alpha_map[L], label=f"{model_name}-{L} (Wrong)")

            plt.title(f"Correct vs Wrong — {metric_name}")
            plt.xlabel("λ")
            plt.ylabel(metric_name)
            plt.grid(alpha=0.3)
            handles, labels = plt.gca().get_legend_handles_labels()
            uniq = dict(zip(labels, handles))
            plt.legend(uniq.values(), uniq.keys(), bbox_to_anchor=(1.05,1), loc="upper left")
            plt.tight_layout()
            plt.savefig(f"{outdir}/{metric_key}.png", dpi=300)
            plt.close()

    # --- Plot B: All Samples ---
    def plot_all():
        metrics = {
            "prefix": "Avg Prefix Length",
            "first_error": "Avg First Error Step",
            "avg_scores": "Avg Step Correctness",
            "avg_steps": "Avg Total Steps",
            "avg_total_tokens": "Avg Total Tokens",
            "avg_tokens_per_step": "Avg Tokens per Step",
            "acc": "Accuracy",
            "total_error_steps": "Total Error Steps",
            "error_step_ratio": "Error Step Ratio"
        }
        outdir = f"{SAVE_ROOT}/all"
        
        for metric_key, metric_name in metrics.items():
            plt.figure(figsize=(10,6))
            for model_name, model_data in model_results.items():
                layers = detect_layers(model_data)
                alpha_map = {L: LAYER_ALPHA[i % len(LAYER_ALPHA)] for i, L in enumerate(layers)}

                for L in layers:
                    lambdas = detect_lambdas(model_data, L)
                    lam_vals, mean_vals = [], []

                    for lam in lambdas:
                        entry = model_data[L][lam]
                        M = extract_metrics(entry, THRESHOLD, model_name)
                        lam_vals.append(lam_to_float(lam))
                        if metric_key == "acc":
                            mean_vals.append(np.mean(M["Y"]))
                        else:
                            mean_vals.append(np.mean(M[metric_key]))

                    idx = np.argsort(lam_vals)
                    plt.plot(np.array(lam_vals)[idx], np.array(mean_vals)[idx],
                             color=MODEL_COLOR_MAP[model_name], linestyle=MODEL_LINESTYLE_MAP[model_name],
                             marker=MODEL_MARKER_MAP[model_name], markersize=6, alpha=alpha_map[L],
                             label=f"{model_name}-{L}")

            plt.title(f"All Samples — {metric_name}")
            plt.xlabel("λ")
            plt.ylabel(metric_name)
            plt.grid(alpha=0.3)
            handles, labels = plt.gca().get_legend_handles_labels()
            uniq = dict(zip(labels, handles))
            plt.legend(uniq.values(), uniq.keys(), bbox_to_anchor=(1.05,1), loc="upper left")
            plt.tight_layout()
            plt.savefig(f"{outdir}/{metric_key}.png", dpi=300)
            plt.close()

    # --- Plot C: Scatters (X vs Accuracy) ---
    def plot_scatter(x_metric_fn, x_label, filename):
        plt.figure(figsize=(10,6))
        seen_models = set()
        for model_name, model_data in model_results.items():
            layers = detect_layers(model_data)
            for L in layers:
                for lam, entry in model_data[L].items():
                    val = x_metric_fn(entry)
                    acc = np.mean(entry["Y"])
                    label = model_name if model_name not in seen_models else None
                    seen_models.add(model_name)
                    plt.scatter(val, acc, color=MODEL_COLOR_MAP[model_name], 
                                marker=MODEL_MARKER_MAP[model_name], s=140, alpha=0.85, label=label)
        
        plt.xlabel(x_label)
        plt.ylabel("Accuracy")
        plt.title(f"{x_label} vs Accuracy")
        plt.grid(alpha=0.3)
        plt.legend(title="Model")
        plt.savefig(f"{SAVE_ROOT}/special/{filename}.png", dpi=300)
        plt.close()

    # --- Plot D: Per Step Avg ---
    def plot_per_step_avg(max_steps=15):
        outdir = f"{SAVE_ROOT}/per_step_avg"
        for model_name, model_data in model_results.items():
            layers = detect_layers(model_data)
            for L in layers:
                lambdas = detect_lambdas(model_data, L)
                for lam in lambdas:
                    entry = model_data[L][lam]
                    step_scores_all = entry["step_scores"]
                    Y = np.array(entry["Y"])
                    
                    step_scores_corr = [s for i, s in enumerate(step_scores_all) if Y[i] == 1]
                    step_scores_wrong = [s for i, s in enumerate(step_scores_all) if Y[i] == 0]

                    mean_all, n_all = per_step_mean(step_scores_all, max_steps)
                    mean_corr, n_corr = per_step_mean(step_scores_corr, max_steps)
                    mean_wrong, n_wrong = per_step_mean(step_scores_wrong, max_steps)

                    if len(mean_all) == 0: continue
                    plt.figure(figsize=(9, 5))
                    
                    plt.plot(np.arange(1, len(mean_all)+1), mean_all, marker="o", lw=2, label=f"All (n@1={n_all[0]})")
                    if len(mean_corr) > 0:
                        plt.plot(np.arange(1, len(mean_corr)+1), mean_corr, marker="s", lw=2, label=f"Correct (n@1={n_corr[0]})")
                    if len(mean_wrong) > 0:
                        plt.plot(np.arange(1, len(mean_wrong)+1), mean_wrong, marker="x", lw=2, label=f"Wrong (n@1={n_wrong[0]})")

                    plt.axhline(THRESHOLD, linestyle="--", lw=1.2, label=f"thr={THRESHOLD}")
                    plt.xlabel("Step k")
                    plt.ylabel("Avg PRM Score")
                    plt.title(f"Per-step Avg Correctness\n{model_name} | L={L} | λ={lam}")
                    plt.grid(alpha=0.25)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f"{outdir}/{_safe_name(model_name)}_L{L}_{_safe_name(lam)}_per_step_avg.png", dpi=300)
                    plt.close()

    # Execute all internal plotting functions
    print(">>> generating correct vs wrong plots...")
    plot_correct_wrong()
    
    print(">>> generating all-sample plots...")
    plot_all()

    print(">>> generating scatter plots...")
    plot_scatter(lambda e: np.mean([np.mean(s) for s in e["step_scores"]]), "Avg Step Score", "avg_step_score_vs_accuracy")
    plot_scatter(lambda e: np.mean([np.mean(s) for s in e["step_token_len"]]), "Avg Tokens per Step", "avg_tokens_vs_accuracy")
    plot_scatter(lambda e: np.mean([np.sum(s) for s in e["step_token_len"]]), "Avg Total Tokens", "avg_total_tokens_vs_accuracy")
    plot_scatter(lambda e: np.mean([len(s) for s in e["step_token_len"]]), "Avg Number of Steps", "avg_num_steps_vs_accuracy")
    plot_scatter(lambda e: np.mean([sum(1 for s in steps if s < THRESHOLD) for steps in e["step_scores"]]), "Avg Error Steps", "avg_error_steps_vs_accuracy")

    print(">>> generating per-step traces...")
    plot_per_step_avg()

# ============================================================
# 5. MAIN EXECUTION FLOW
# ============================================================
if __name__ == "__main__":
    # ---------------------------
    # 0. ARGUMENT PARSING
    # ---------------------------
    parser = argparse.ArgumentParser(description="Run evaluation pipeline.")
    parser.add_argument(
        "--plot-only", 
        action="store_true", 
        help="Skip GPU jobs and merging, directly run analysis plots on existing merged data."
    )
    args = parser.parse_args()

    # Define the path for the merged result file (used in multiple phases)
    merged_path = f"{FOLDER}/results_merged.json"

    # ---------------------------
    # EXECUTION CONTROL
    # ---------------------------
    if not args.plot_only:
        # ==========================================
        # RUN FULL PIPELINE (PHASE 1 - 3)
        # ==========================================
        
        # ---------------------------
        # PHASE 1: PREPARE JOBS
        # ---------------------------
        jobs = []
        print(f"Scanning for job files in {BASE_DIR}...")
        
        if os.path.exists(BASE_DIR):
            for entry in os.listdir(BASE_DIR):
                full_path = os.path.join(BASE_DIR, entry)
                if not os.path.isdir(full_path):
                    continue
                
                parts = entry.split('_')
                if len(parts) < 3:
                    continue
                    
                lam_str = parts[-1]
                layer_str = parts[-2]
                
                if not (layer_str.startswith('L') and layer_str[1:].isdigit()):
                    continue
                    
                model_name = "_".join(parts[:-2])
                
                subdirs = [d for d in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, d))]
                model_subdir = next((d for d in subdirs if d.startswith("Qwen__")), None)
                
                if not model_subdir:
                    if subdirs:
                        model_subdir = subdirs[0]
                    else:
                        continue
                
                search_path = os.path.join(full_path, model_subdir)
                pattern = os.path.join(search_path, "samples_*.jsonl")
                files = sorted(glob.glob(pattern))
                
                if not files:
                    continue
                
                jsonl_file = files[-1]
                
                if model_name in MODEL_MAP:
                    gen_model_name = MODEL_MAP[model_name]
                else:
                    gen_model_name = f"Qwen/{model_name}"
                
                jobs.append((model_name, gen_model_name, layer_str, lam_str, jsonl_file))

        print(f"Total jobs to run = {len(jobs)}")

        # ---------------------------
        # PHASE 2: RUN WORKERS
        # ---------------------------
        if jobs:
            pool = mp.Pool(NUM_GPUS)
            for idx, job in enumerate(jobs):
                gpu_id = GPU_IDS[idx % NUM_GPUS]
                pool.apply_async(worker, args=(idx, job, gpu_id))

            pool.close()
            pool.join()
            print("✔ All GPU jobs finished.")
        else:
            print("No jobs found to run. Checking for existing merged results...")

        # ---------------------------
        # PHASE 3: MERGE RESULTS
        # ---------------------------
        final = {}
        chunk_files = sorted(glob.glob(f"{FOLDER}/results_chunk_*.json"))
        
        if chunk_files:
            print(f"Merging {len(chunk_files)} chunk files...")
            for f in chunk_files:
                try:
                    part = json.load(open(f))
                    for model in part:
                        final.setdefault(model, {})
                        for L in part[model]:
                            final[model].setdefault(L, {})
                            for lam in part[model][L]:
                                final[model][L][lam] = part[model][L][lam]
                except Exception as e:
                    print(f"Error reading {f}: {e}")

            with open(merged_path, "w") as f:
                json.dump(final, f, indent=2)
            print(f"🎉 Final merged JSON saved to: {merged_path}")
        else:
            if os.path.exists(merged_path):
                 print(f"Found existing merged file at: {merged_path}")
            else:
                 print("❌ No chunk files found and no merged file exists. Exiting.")
                 exit()

    else:
        print(f"⏩ Skipping Scan/Worker/Merge phases (Plot Only Mode).")
        print(f"   Targeting merged file: {merged_path}")

    # ---------------------------
    # PHASE 4: RUN ANALYSIS PLOTS
    # ---------------------------
    # This phase runs if:
    # 1. We just finished the full pipeline OR
    # 2. We used --plot-only and the file exists
    
    if not os.path.exists(merged_path):
        print(f"❌ Critical: Merged result file not found at {merged_path}. Cannot plot.")
        exit(1)

    print("\n" + "="*30)
    print("STARTING ANALYSIS & PLOTTING")
    print("="*30)

    try:
        model_results = {}
        raw_data = json.load(open(merged_path))
        model_results = raw_data
        
        # Trigger plotting
        run_all_plots(model_results)
        print("\n✅ All plots generated successfully.")
        
    except Exception as e:
        print(f"❌ Error during plotting phase: {e}")
        import traceback
        traceback.print_exc()