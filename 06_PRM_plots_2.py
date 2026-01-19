import os
import json
import glob
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 1. GLOBAL CONFIGURATION & PATHS
# ============================================================

# User Paths
PREFIX = "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_less_tokens_3B_lib_manual_same_lib_1000samples"
FOLDER = os.path.join(PREFIX, "prm_results")
BASE_DIR = os.path.join(PREFIX, "gsm8k_cot_zeroshot")
PRM_MODEL = "Qwen/Qwen2.5-Math-PRM-7B"

# Set to TRUE to enable expensive model-based metrics (PPL, KL, Rank Shift)
# This will try to load the model on GPU.
ENABLE_PRM_SCORING = True
ENABLE_MODEL_METRICS = False
ENABLE_PLOTTING = True

os.makedirs(FOLDER, exist_ok=True)
SAVE_ROOT = FOLDER
for sub in ["correct_wrong", "scatter", "special", "all", "per_step_avg", "reasoning_errors", "advanced_stats"]:
    os.makedirs(os.path.join(SAVE_ROOT, sub), exist_ok=True)

NUM_GPUS = 8
GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]

STEP = 0.5
STEER_LAMBDAS = [i * STEP for i in range(-5, 6)]
STEER_LAMBDAS = [0.0]

def lam_to_str(lam: float) -> str:
    if abs(lam) < 1e-9:
        return "BASELINE"
    sign = "-" if lam < 0 else ""
    s = f"{abs(lam):.2f}".rstrip("0")
    if s.endswith("."): s += "0"
    s = s.replace(".", "p")
    return f"lam{sign}{s}"

lam_values = [lam_to_str(lam) for lam in STEER_LAMBDAS]

MODEL_MAP = {
    "Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
}

MODEL_TO_LAYERS = {
    "Qwen2.5-3B-Instruct": [1],
}

# ============================================================
# 2. ADVANCED METRIC CALCULATORS
# ============================================================

class ReasoningAnalyzer:
    """Calculates heuristics for reasoning quality based on PRM scores and text."""
    
    @staticmethod
    def detect_repetition(steps_text):
        """Detects repetition loops using unique n-gram ratios or step similarity."""
        if not steps_text: return 0.0
        # Simple heuristic: Ratio of unique steps to total steps
        unique_steps = set([s.strip() for s in steps_text])
        return 1.0 - (len(unique_steps) / len(steps_text))

    @staticmethod
    def detect_logical_leap(step_scores, threshold_drop=0.5):
        """
        Detects 'Logical Leap': A high-confidence step followed immediately 
        by a very low-confidence step.
        """
        leaps = 0
        for i in range(len(step_scores) - 1):
            curr = step_scores[i]
            next_s = step_scores[i+1]
            # If we were confident (>0.8) and suddenly dropped (>0.5 drop)
            if curr > 0.8 and (curr - next_s) > threshold_drop:
                leaps += 1
        return leaps

    @staticmethod
    def detect_context_forgetfulness(step_scores, threshold=0.72):
        """
        Detects 'Context Forgetfulness': A long chain of correct steps 
        that suddenly fails at the very end (late error).
        """
        if len(step_scores) < 3: return 0
        
        # Check if the first 70% of steps are good
        split_idx = int(len(step_scores) * 0.7)
        early_steps = step_scores[:split_idx]
        late_steps = step_scores[split_idx:]
        
        if np.mean(early_steps) > threshold and np.mean(late_steps) < threshold:
            return 1 # Potential forgetfulness
        return 0

    @staticmethod
    def calculate_text_similarity(gen_text, ref_text):
        """Cosine similarity of TF-IDF vectors."""
        if not gen_text or not ref_text: return 0.0
        try:
            tfidf = TfidfVectorizer().fit_transform([gen_text, ref_text])
            return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        except:
            return 0.0

class ModelEvaluator:
    """Handles expensive PPL / Rank Shift calculations requiring the model."""
    def __init__(self, model_name, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        self.device = device
        self.loss_fct = CrossEntropyLoss(reduction="none")

    def compute_ppl_and_rank(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, :-1, :]
            labels = inputs.input_ids[:, 1:]
            
            # Perplexity
            shift_logits = logits.reshape(-1, logits.size(-1))
            shift_labels = labels.reshape(-1)
            loss = self.loss_fct(shift_logits, shift_labels)
            ppl = torch.exp(loss.mean()).item()
            
            # Rank (Simplistic Rank Shift Metric)
            # Rank of the true token in the predicted distribution
            probs = torch.softmax(logits, dim=-1)
            # Get rank of the ground truth label
            sorted_idxs = torch.argsort(probs, dim=-1, descending=True)
            ranks = (sorted_idxs == labels.unsqueeze(-1)).nonzero(as_tuple=True)[-1] + 1
            avg_rank = ranks.float().mean().item()
            
        return ppl, avg_rank

# ============================================================
# 3. METRIC EXTRACTION
# ============================================================

THRESHOLD = 0.9
MODEL_MARKERS = ["o", "s", "^", "D", "P", "X", "<", ">", "v"]
WRONG_MARKERS = ["x", "X", "v", "p", "*", "d", "h", "H", "+"]
MODEL_LINESTYLES = ["-", "--", "-.", ":"]
LAYER_ALPHA = [1.0, 0.7, 0.5, 0.2]

def lam_to_float(lam):
    if lam == "BASELINE": return 0.0
    return float(lam[3:].replace("p", "."))

def detect_layers(model_data): return list(model_data.keys())
def detect_lambdas(model_data, L): return list(model_data[L].keys())
def _safe_name(s): return s.replace("/", "_").replace(" ", "_").replace(":", "_").replace("__", "_")

def compute_prefix_first_error(step_scores, thr):
    prefix_list, fe_list = [], []
    for scores in step_scores:
        prefix, fe = 0, None
        for i, s in enumerate(scores):
            if s >= thr: prefix += 1
            else:
                fe = i + 1; break
        if fe is None: fe = len(scores) + 1
        prefix_list.append(prefix); fe_list.append(fe)
    return np.array(prefix_list), np.array(fe_list)

def extract_metrics(entry, thr, model_name=None):
    step_scores = entry["step_scores"]
    step_token_len = entry["step_token_len"]
    Y = np.array(entry["Y"])
    
    # Try to get text if available (Assuming entry has 'samples' or similar structure)
    # This part depends on your JSONL structure. Assuming 'steps_text' exists.
    # If not, some heuristics (like Repetition) will be 0.
    steps_text_list = entry.get("steps_text", [[]]*len(Y)) 
    full_text_list = entry.get("generated_text", [""]*len(Y))
    gold_solution_list = entry.get("solution", [""]*len(Y))

    prefix, first_err = compute_prefix_first_error(step_scores, thr)

    avg_scores = np.array([np.mean(s) if len(s)>0 else 0 for s in step_scores])
    avg_steps = np.array([len(s) for s in step_scores])
    avg_total_tokens = np.array([np.sum(lens) if len(lens)>0 else 0 for lens in step_token_len])
    
    # --- NEW METRICS ---
    
    # 1. Repetition Loops
    repetition_scores = np.array([ReasoningAnalyzer.detect_repetition(s) for s in steps_text_list])
    
    # 2. Logical Leaps
    logical_leaps = np.array([ReasoningAnalyzer.detect_logical_leap(s) for s in step_scores])
    
    # 3. Context Forgetfulness (Late Error)
    forgetfulness = np.array([ReasoningAnalyzer.detect_context_forgetfulness(s) for s in step_scores])

    # 4. Text Similarity (if gold solution exists)
    text_sim = []
    if len(full_text_list) > 0 and len(gold_solution_list) > 0:
        for gen, ref in zip(full_text_list, gold_solution_list):
            text_sim.append(ReasoningAnalyzer.calculate_text_similarity(gen, ref))
    text_sim = np.array(text_sim) if text_sim else np.zeros_like(Y, dtype=float)

    # 5. Placeholders for Model-Based Metrics (PPL, Rank)
    ppl_scores = np.array(entry.get("ppl", []))
    if len(ppl_scores) == 0:
        ppl_scores = np.zeros_like(Y, dtype=float)

    rank_scores = np.array(entry.get("rank_shift", []))
    if len(rank_scores) == 0:
        rank_scores = np.zeros_like(Y, dtype=float)

    return {
        "prefix": prefix,
        "first_error": first_err,
        "avg_scores": avg_scores,
        "avg_steps": avg_steps,
        "avg_total_tokens": avg_total_tokens,
        "repetition": repetition_scores,
        "logical_leaps": logical_leaps,
        "forgetfulness": forgetfulness,
        "text_similarity": text_sim,
        "ppl": ppl_scores,
        "rank_shift": rank_scores,
        "Y": Y
    }

def per_step_mean(step_scores, max_steps=None):
    if len(step_scores) == 0: return np.array([]), np.array([])
    lens = np.array([len(s) for s in step_scores], dtype=int)
    K = int(min(max_steps, lens.max())) if max_steps else int(lens.max())
    means = np.full(K, np.nan, dtype=float)
    counts = np.zeros(K, dtype=int)
    for k in range(K):
        vals = [s[k] for s in step_scores if len(s) > k]
        counts[k] = len(vals)
        if counts[k] > 0: means[k] = float(np.mean(vals))
    return means, counts

# ============================================================
# 4. WORKER FUNCTION
# ============================================================

def worker(job_idx, job, gpu_id):
    model_name, gen_model_name, L, lam, jsonl = job
    out_file = f"{FOLDER}/results_chunk_{job_idx}.json"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 1. Run PRM Scoring (Existing Logic)
    if ENABLE_PRM_SCORING:
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
    
    # 2. (Optional) Run Expensive Model Metrics
    # This requires modifying the chunk file in place or creating a sidecar file.
    if ENABLE_MODEL_METRICS:
        print(f"[GPU{gpu_id}] Computing PPL/Rank for {jsonl}...")
        try:
            if not os.path.exists(out_file):
                print(f"[GPU{gpu_id}] Output file {out_file} not found. Skipping metrics.")
                return

            with open(out_file, 'r') as f:
                data = json.load(f)

            # Locate the entry in the potentially nested dictionary
            # data structure: {model: {L: {lam: entry}}}
            if model_name in data and L in data[model_name] and lam in data[model_name][L]:
                entry = data[model_name][L][lam]
                texts = entry.get("generated_text", [])
                
                if texts:
                    evaluator = ModelEvaluator(gen_model_name, device=f"cuda:{gpu_id}")
                    ppl_list = []
                    rank_list = []
                    
                    for text in texts:
                        ppl, rank = evaluator.compute_ppl_and_rank(text)
                        ppl_list.append(ppl)
                        rank_list.append(rank)
                        
                    entry["ppl"] = ppl_list
                    entry["rank_shift"] = rank_list
                    
                    # Save back to file
                    with open(out_file, 'w') as f:
                        json.dump(data, f, indent=2)
                    print(f"[GPU{gpu_id}] Added PPL/Rank metrics to {out_file}")
            else:
                print(f"[GPU{gpu_id}] Could not find entry in {out_file}")

        except Exception as e:
            print(f"Failed model metrics: {e}")

# ============================================================
# 5. PLOTTING FUNCTIONS
# ============================================================

def run_all_plots(model_results):
    all_models = list(model_results.keys())
    print(f"Generating plots for models: {all_models}")

    color_list = plt.cm.tab20(np.linspace(0, 1, len(all_models)))
    MODEL_COLOR_MAP = {model: color_list[i] for i, model in enumerate(all_models)}
    MODEL_MARKER_MAP = {model: MODEL_MARKERS[i % len(MODEL_MARKERS)] for i, model in enumerate(all_models)}
    MODEL_LINESTYLE_MAP = {model: MODEL_LINESTYLES[i % len(MODEL_LINESTYLES)] for i, model in enumerate(all_models)}

    # --- Plot A: Generic Line Plotter ---
    def generic_plot(metric_key, title, ylabel, subdir):
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
                    mean_vals.append(np.mean(M[metric_key]))

                idx = np.argsort(lam_vals)
                plt.plot(np.array(lam_vals)[idx], np.array(mean_vals)[idx],
                            color=MODEL_COLOR_MAP[model_name], linestyle=MODEL_LINESTYLE_MAP[model_name],
                            marker=MODEL_MARKER_MAP[model_name], markersize=6, alpha=alpha_map[L],
                            label=f"{model_name}-{L}")

        plt.title(title)
        plt.xlabel("Steering vector multiplier (λ)")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"{SAVE_ROOT}/{subdir}/{metric_key}.png", dpi=300)
        plt.close()

    # --- Plot Group 1: Standard Metrics ---
    metrics_std = {
        "prefix": ("Avg Prefix Length", "Steps"),
        "first_error": ("Avg First Error Step", "Step Index"),
        "avg_scores": ("Avg Step Correctness", "PRM Score"),
        "avg_total_tokens": ("Avg Total Tokens", "Count")
    }
    for k, (tit, ylab) in metrics_std.items():
        generic_plot(k, tit, ylab, "all")

    # --- Plot Group 2: Reasoning Errors & Text Stats ---
    print(">>> generating reasoning error plots...")
    metrics_adv = {
        "repetition": ("Repetition Ratio (Looping)", "Repetition Score (0-1)"),
        "logical_leaps": ("Avg Logical Leaps per Sample", "Count"),
        "forgetfulness": ("Context Forgetfulness Rate", "Rate"),
        "text_similarity": ("Text Similarity to Gold", "Cosine Sim")
    }
    for k, (tit, ylab) in metrics_adv.items():
        generic_plot(k, tit, ylab, "reasoning_errors")

    # --- Plot Group 3: Scatters ---
    def plot_scatter(x_metric_fn, x_label, filename):
        plt.figure(figsize=(10,6))
        seen_models = set()
        for model_name, model_data in model_results.items():
            layers = detect_layers(model_data)
            for L in layers:
                for lam, entry in model_data[L].items():
                    M = extract_metrics(entry, THRESHOLD)
                    val = np.mean(M[x_label]) if isinstance(x_label, str) else x_metric_fn(M)
                    acc = np.mean(M["Y"])
                    label = model_name if model_name not in seen_models else None
                    seen_models.add(model_name)
                    plt.scatter(val, acc, color=MODEL_COLOR_MAP[model_name], 
                                marker=MODEL_MARKER_MAP[model_name], s=140, alpha=0.85, label=label)
        
        plt.xlabel(x_label if isinstance(x_label, str) else "Metric")
        plt.ylabel("Accuracy")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(f"{SAVE_ROOT}/scatter/{filename}.png", dpi=300)
        plt.close()

    print(">>> generating scatter plots...")
    plot_scatter("repetition", "Repetition Score", "repetition_vs_accuracy")
    plot_scatter("text_similarity", "Text Similarity", "similarity_vs_accuracy")
    plot_scatter("avg_scores", "Avg PRM Score", "prm_score_vs_accuracy")

    # --- Plot Group 4: Per Step Avg ---
    print(">>> generating per-step traces...")
    outdir = f"{SAVE_ROOT}/per_step_avg"
    for model_name, model_data in model_results.items():
        layers = detect_layers(model_data)
        for L in layers:
            lambdas = detect_lambdas(model_data, L)
            for lam in lambdas:
                entry = model_data[L][lam]
                step_scores_all = entry["step_scores"]
                Y = np.array(entry["Y"])
                
                mean_all, _ = per_step_mean(step_scores_all, 15)
                mean_corr, _ = per_step_mean([s for i,s in enumerate(step_scores_all) if Y[i]==1], 15)
                mean_wrong, _ = per_step_mean([s for i,s in enumerate(step_scores_all) if Y[i]==0], 15)

                if len(mean_all) == 0: continue
                plt.figure(figsize=(9, 5))
                plt.plot(np.arange(1, len(mean_all)+1), mean_all, marker="o", label="All")
                if len(mean_corr)>0: plt.plot(np.arange(1, len(mean_corr)+1), mean_corr, marker="s", label="Correct")
                if len(mean_wrong)>0: plt.plot(np.arange(1, len(mean_wrong)+1), mean_wrong, marker="x", label="Wrong")
                
                plt.axhline(THRESHOLD, linestyle="--", color='black', alpha=0.5)
                plt.title(f"Step-wise Correctness Rate\n{model_name} {L} {lam}")
                plt.ylabel("Avg PRM Score")
                plt.xlabel("Step")
                plt.legend()
                plt.savefig(f"{outdir}/{_safe_name(model_name)}_L{L}_{_safe_name(lam)}.png", dpi=300)
                plt.close()

# ============================================================
# 6. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    jobs = []
    print(f"Scanning for job files in {BASE_DIR}...")
    
    if os.path.exists(BASE_DIR):
        for entry in os.listdir(BASE_DIR):
            full_path = os.path.join(BASE_DIR, entry)
            if not os.path.isdir(full_path): continue
            
            parts = entry.split('_')
            if len(parts) < 3: continue
            
            lam_str = parts[-1]
            layer_str = parts[-2]
            if not (layer_str.startswith('L') and layer_str[1:].isdigit()): continue
            
            model_name = "_".join(parts[:-2])
            subdirs = [d for d in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, d))]
            model_subdir = next((d for d in subdirs if d.startswith("Qwen__")), subdirs[0] if subdirs else None)
            
            if not model_subdir: continue
            search_path = os.path.join(full_path, model_subdir)
            files = sorted(glob.glob(os.path.join(search_path, "samples_*.jsonl")))
            
            if not files: continue
            jsonl_file = files[-1]
            gen_model_name = MODEL_MAP.get(model_name, f"Qwen/{model_name}")
            jobs.append((model_name, gen_model_name, layer_str, lam_str, jsonl_file))

    print(f"Total jobs to run = {len(jobs)}")

    if jobs and (ENABLE_PRM_SCORING or ENABLE_MODEL_METRICS):
        pool = mp.Pool(NUM_GPUS)
        for idx, job in enumerate(jobs):
            gpu_id = GPU_IDS[idx % NUM_GPUS]
            pool.apply_async(worker, args=(idx, job, gpu_id))
        pool.close()
        pool.join()
        print("✔ All GPU jobs finished.")

    # Merge Results
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
            except Exception as e: print(f"Error {f}: {e}")
        
        merged_path = f"{FOLDER}/results_merged.json"
        with open(merged_path, "w") as f: json.dump(final, f, indent=2)
    else:
        merged_path = f"{FOLDER}/results_merged.json"
        if not os.path.exists(merged_path):
             print("❌ No results found. Exiting."); exit()

    print("\n>>> STARTING PLOTTING")
    if ENABLE_PLOTTING:
        try:
            model_results = json.load(open(merged_path))
            run_all_plots(model_results)
            print("\n✅ All plots generated.")
        except Exception as e:
            print(f"❌ Plotting error: {e}")
        import traceback; traceback.print_exc()