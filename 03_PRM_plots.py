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
import time

# Import plotting functions from 05_plots_concise
import sys
# Assuming 05_plots_concise.py is in the same directory, import it
# If it's not in the PYTHONPATH, we might need to add it, but it seems to be adjacent.
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("plots_concise", "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/05_plots_concise.py")
    plots_concise = importlib.util.module_from_spec(spec)
    sys.modules["plots_concise"] = plots_concise
    spec.loader.exec_module(plots_concise)
except ImportError:
    # Fallback if standard import works
    import plots_concise

# ============================================================
# 1. GLOBAL CONFIGURATION & PATHS
# ============================================================

REWRITE_PATH = "gpt_rewrites_unified"
# User Paths
PROMPT_STYLE = "expert_leap"  # "old" or "expert_leap"
TASK = "gsm8k_cot_zeroshot_unified" # "gsm8k_cot_zeroshot_unified"
# PREFIX = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/gpt_rewrites_unified/Qwen_Qwen2.5-3B-Instruct/vectors_50_old/Qwen_Qwen2.5-3B-Instruct_applied/"
PREFIX = f"/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/{REWRITE_PATH}/Qwen_Qwen2.5-3B-Instruct/vectors_50_{PROMPT_STYLE}/Qwen_Qwen2.5-3B-Instruct_applied/"

FOLDER = os.path.join(PREFIX, "prm_results")
BASE_DIR = os.path.join(PREFIX, TASK)
PRM_MODEL = "Qwen/Qwen2.5-Math-PRM-7B"

# Set to TRUE to enable expensive model-based metrics (PPL, KL, Rank Shift)
# This will try to load the model on GPU.
ENABLE_PRM_SCORING = True
ENABLE_MODEL_METRICS = True
ENABLE_PLOTTING = True

os.makedirs(FOLDER, exist_ok=True)
SAVE_ROOT = FOLDER
for sub in ["correct_wrong", "scatter", "special", "all", "per_step_avg", "reasoning_errors", "advanced_stats"]:
    os.makedirs(os.path.join(SAVE_ROOT, sub), exist_ok=True)

NUM_GPUS = 8
GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]

MODEL_MAP = {
    "Qwen2.5-32B-Instruct": "Qwen/Qwen2.5-32B-Instruct",
}


STATUS_FILE = os.path.join(FOLDER, "job_status.json")

def load_status():
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_status_update(key, status):
    data = load_status()
    data[key] = status
    with open(STATUS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

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
    def detect_logical_leap(step_scores, threshold_drop=0.7):
        """
        Detects 'Logical Leap': A high-confidence step followed immediately 
        by a very low-confidence step.
        """
        leaps = 0
        for i in range(len(step_scores) - 1):
            curr = step_scores[i]
            next_s = step_scores[i+1]
            # If we were confident (>0.9) and suddenly dropped (>0.7 drop)
            if curr > 0.9 and (curr - next_s) > threshold_drop:
                leaps += 1
        return leaps

    @staticmethod
    def detect_context_forgetfulness(step_scores, threshold=0.9):
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
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelEvaluator:
    """Handles expensive PPL / Rank Shift calculations requiring the model."""
    def __init__(self, base_model_name, device="cuda"):
        # 注意：这里加载的必须是【微调前的 Base Model】
        # 因为我们要看微调后的输出在 Base Model 眼中有多"意外"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name, 
            torch_dtype=torch.float16
        ).to(device)
        self.device = device
        self.loss_fct = CrossEntropyLoss(reduction="none")
        self.model.eval()

    def compute_ppl_and_rank(self, text):
        """保留原本的 PPL 计算逻辑"""
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
            
            # Rank (平均 Rank)
            probs = torch.softmax(logits, dim=-1)
            sorted_idxs = torch.argsort(probs, dim=-1, descending=True)
            ranks = (sorted_idxs == labels.unsqueeze(-1)).nonzero(as_tuple=True)[-1] + 1
            avg_rank = ranks.float().mean().item()
            
        return ppl, avg_rank

    def compute_rank_shift(self, prompt, full_response):
        """
        计算 Rank Shift：微调后生成的 Token 在 Base Model 中的排名。
        返回所有 Token 的 Rank 列表，以便后续筛选 Top-k Shifted Tokens。
        """
        # 1. 分别 Tokenize 并拼接，确保边界清晰
        # Prompt 通常会有 BOS token (如果是句子开头)
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        # Response 不加 BOS，因为它是接在 Prompt 后面的
        response_inputs = self.tokenizer(full_response, return_tensors="pt", add_special_tokens=False)
        
        prompt_ids = prompt_inputs.input_ids.to(self.device)
        response_ids = response_inputs.input_ids.to(self.device)
        
        # 拼接 Input IDs
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        
        # 确定 Response 开始的索引
        start_idx = prompt_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
            
            # 2. 截取 Response 部分的 Logits 和 Labels
            # Logits[i] 预测的是 input_ids[i+1]
            # 我们要预测从 start_idx 开始的 Token (即 Response 的第一个词)
            # 对应的 Logit 是 start_idx - 1
            
            # shift_logits: [1, response_len, vocab_size]
            shift_logits = logits[:, start_idx-1 : -1, :]
            # shift_labels: [1, response_len]
            shift_labels = input_ids[:, start_idx:]
            
            # 3. 计算每个 Token 的 Rank
            # 为了加速，我们可以只计算目标 Token 的 Rank，而不是全排序
            # Rank = (有多少个词的分数 > 目标词的分数) + 1
            
            token_ranks = []
            
            # 遍历 Response 中的每个 Token
            for i in range(shift_labels.shape[1]):
                target_id = shift_labels[0, i].item()
                current_logits = shift_logits[0, i]
                
                # 获取目标 Token 的 Logit 值
                target_score = current_logits[target_id]
                
                # 计算 Rank: 比 target_score 大的 Logit 数量 + 1
                rank = (current_logits > target_score).sum().item() + 1
                
                token_str = self.tokenizer.decode([target_id])
                
                token_ranks.append({
                    "token": token_str,
                    "rank": rank,
                    "id": target_id
                })
                
        return token_ranks

    def find_most_shifted_tokens(self, token_ranks, top_k=5):
        """
        根据 Rank 大小筛选出 Shift 最大的 Token
        """
        # 按照 Rank 降序排列 (Rank 越大，说明 Base Model 越意想不到)
        sorted_ranks = sorted(token_ranks, key=lambda x: x['rank'], reverse=True)
        return sorted_ranks[:top_k]


# ============================================================
# 4. WORKER FUNCTION
# ============================================================

def worker(job_idx, job, gpu_id):
    model_name, gen_model_name, L, lam, jsonl = job
    out_file = f"{FOLDER}/results_chunk_{gen_model_name}_{L}_{lam}.json"
    job_key = f"{gen_model_name}_{L}_{lam}"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    success = True

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
        ret = os.system(cmd)
        if ret != 0:
            success = False
    
    # 2. (Optional) Run Expensive Model Metrics
    # This requires modifying the chunk file in place or creating a sidecar file.
    if success and ENABLE_MODEL_METRICS:
        print(f"[GPU{gpu_id}] Computing PPL/Rank for {jsonl}...")
        try:
            if not os.path.exists(out_file):
                print(f"[GPU{gpu_id}] Output file {out_file} not found. Skipping metrics.")
                success = False
            else:
                with open(out_file, 'r') as f:
                    data = json.load(f)

            # Locate the entry in the potentially nested dictionary
            # data structure: {model: {L: {lam: entry}}}
            if model_name in data and L in data[model_name] and lam in data[model_name][L]:
                entry = data[model_name][L][lam]
                texts = entry.get("generated_text", [])
                
                if texts:
                    # Load original JSONL to retrieve Prompts
                    try:
                        with open(jsonl, 'r') as f_jsonl:
                            raw_data = [json.loads(line) for line in f_jsonl]
                    except Exception as e:
                        print(f"[GPU{gpu_id}] Failed to read jsonl {jsonl}: {e}")
                        raw_data = []

                    evaluator = ModelEvaluator(gen_model_name, device=f"cuda:0")
                    ppl_list = []
                    rank_list = []
                    token_ranks_list = []
                    most_shifted_list = []
                    
                    data_idx = 0

                    for text in texts:
                        ppl, rank = evaluator.compute_ppl_and_rank(text)
                        ppl_list.append(ppl)
                        rank_list.append(rank)
                        
                        # Find corresponding prompt
                        prompt = ""
                        for j in range(data_idx, len(raw_data)):
                            d = raw_data[j]
                            if d.get("filter") == "strict-match": continue
                            try:
                                cot_cand = d["resps"][0][0].strip()
                            except: continue

                            if cot_cand == text:
                                prompt = d.get("arguments", {}).get("gen_args_0", {}).get("arg_0", "")
                                data_idx = j + 1
                                break
                        
                        if prompt:
                            tr = evaluator.compute_rank_shift(prompt, text)
                            # token_ranks_list.append(tr)
                            
                            # Find most shifted tokens
                            mst = evaluator.find_most_shifted_tokens(tr, top_k=20)
                            most_shifted_list.append(mst)
                        else:
                            # token_ranks_list.append([])
                            most_shifted_list.append([])
                        
                    entry["ppl"] = ppl_list
                    entry["rank_shift"] = rank_list
                    entry["token_rank_shifts"] = token_ranks_list
                    entry["most_shifted_tokens"] = most_shifted_list
                    
                    # Save back to file
                    with open(out_file, 'w') as f:
                        json.dump(data, f, indent=2)
                    print(f"[GPU{gpu_id}] Added PPL/Rank metrics to {out_file}")
            else:
                print(f"[GPU{gpu_id}] Could not find entry in {out_file}")

        except Exception as e:
            print(f"Failed model metrics: {e}")
            success = False
    
    return job_key, success

# ============================================================
# 5. PLOTTING DELEGATION
# ============================================================

def run_all_plots(model_results):
    print(">>> Delegating to 05_plots_concise.py for plotting...")
    
    # We must call setup_plotting to initialize globals in 05_plots_concise
    plots_concise.setup_plotting(model_results, SAVE_ROOT)
    
    print("\n=== Generating Correct-vs-Wrong plots ===")
    plots_concise.plot_correct_wrong()

    print("\n=== Generating All-sample plots ===")
    plots_concise.plot_all()

    plots_concise.plot_avg_score_vs_acc()
    plots_concise.plot_avg_tokens_vs_acc()
    plots_concise.plot_avg_total_tokens_vs_acc()
    plots_concise.plot_avg_steps_vs_acc()
    plots_concise.plot_avg_error_steps_vs_acc()

    # plots_concise.plot_per_step_avg_correctness(max_steps=15)
    
    print(">>> Plotted via 05_plots_concise.py")

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

    status_map = load_status()
    jobs_to_run = []
    
    for j in jobs:
        # j = (model_name, gen_model_name, layer_str, lam_str, jsonl_file)
        key = f"{j[1]}_{j[2]}_{j[3]}"
        if status_map.get(key) != "success":
            jobs_to_run.append(j)
        else:
            print(f"Skipping {key} (already success)")
            
    jobs = jobs_to_run
    print(f"Jobs to run after filtering = {len(jobs)}")

    if jobs and (ENABLE_PRM_SCORING or ENABLE_MODEL_METRICS):
        pool = mp.Pool(NUM_GPUS)
        
        def update_status_cb(res):
            key, is_success = res
            status_str = "success" if is_success else "failed"
            save_status_update(key, status_str)

        for idx, job in enumerate(jobs):
            gpu_id = GPU_IDS[idx % NUM_GPUS]
            pool.apply_async(worker, args=(idx, job, gpu_id), callback=update_status_cb)
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

    # merged_path = f"{FOLDER}/results_merged.json"
    # if not os.path.exists(merged_path):
    #     print("❌ No results found. Exiting."); exit()

    print("\n>>> STARTING PLOTTING")
    if ENABLE_PLOTTING:
        try:
            model_results = json.load(open(merged_path))
            run_all_plots(model_results)
            print("\n✅ All plots generated.")
        except Exception as e:
            print(f"❌ Plotting error: {e}")
        import traceback; traceback.print_exc()