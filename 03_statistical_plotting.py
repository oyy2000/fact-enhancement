import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import scan_and_process

# ============================================================
# 1. 配置区域 (Configuration)
# ============================================================

STATUS = "gpt_rewrite" # or "large_model_rewrite"

PROMPT_STYLE = "old" # or "old" 
TASKS = "gsm8k_cot_zeroshot_unified_selected_layers"

REWRITE_MODEL = "Qwen/Qwen2.5-3B-Instruct".replace("/", "_") # "meta-llama/Llama-3.1-3B-Instruct".replace("/", "_") #"Qwen/Qwen2.5-7B-Instruct".replace("/", "_")
TARGET_MODEL = "Qwen/Qwen2.5-3B-Instruct".replace("/", "_") #"meta-llama/Llama-3.2-3B-Instruct".replace("/", "_") #"Qwen/Qwen2.5-3B-Instruct".replace("/", "_")

if STATUS == "large_model_rewrite":
    FOLDER_NAME = "large_model_rewrites_unified_new"
    VECTOR_PATH = f"vectors_50_paired_{REWRITE_MODEL}"
elif STATUS == "gpt_rewrite":
    FOLDER_NAME = "gpt_rewrites_unified_new"
    VECTOR_PATH = f"vectors_50_{PROMPT_STYLE}"
DEFAULT_BASE_DIR = f"./{FOLDER_NAME}/{TARGET_MODEL}/{VECTOR_PATH}/{TARGET_MODEL}_applied/"
DEFAULT_OUT_DIR = os.path.join(DEFAULT_BASE_DIR, TASKS, "aaresults/stats_output")

# 绘图配置
PLOT_CONFIG = {
    "acc": {"title": "Accuracy vs Lambda", "ylabel": "Accuracy"},
    "avg_steps": {"title": "Avg Steps vs Lambda", "ylabel": "Average Steps"},
    "avg_tokens_per_step": {"title": "Avg Tokens per Step vs Lambda", "ylabel": "Tokens / Step"},
    "avg_total_tokens": {"title": "Avg Total Tokens vs Lambda", "ylabel": "Total Tokens"},
}


# ============================================================
# 4. 核心逻辑: 绘图 (Plotter)
# ============================================================

def lam_to_float(lam):
    """将 lam1, lam0p5, BASELINE 转为 float"""
    if isinstance(lam, (int, float)): return float(lam)
    if "BASELINE" in lam.upper(): return 0.0
    # 移除 'lam' 前缀
    s = lam.lower().replace("lam", "")
    # 替换 p 为 .
    s = s.replace("p", ".")
    try:
        return float(s)
    except:
        return -1.0

def robust_ylim(values, low=5, high=95, margin=0.1):
    """
    values: list or np.array
    low/high: percentile
    margin: extra headroom ratio
    """
    if len(values) == 0:
        return None

    lo = np.percentile(values, low)
    hi = np.percentile(values, high)
    pad = (hi - lo) * margin if hi > lo else 1e-6
    return lo - pad, hi + pad


all_y_vals = []

def plot_results(results, save_dir, filename_prefix="plot_"):
    """读取 Aggregated Results 并画图"""
    if not results:
        print("No results to plot.")
        return

    os.makedirs(save_dir, exist_ok=True)
    
    metrics = ["acc", "avg_steps", "avg_tokens_per_step", "avg_total_tokens"]
    # 预定义颜色循环
    colors = plt.cm.tab10.colors
    
    for metric in metrics:
        # Check for Acc (common) vs others (split by strategy)
        strategies = ["double", "single"] if metric != "acc" else ["main"]
        
        for strategy in strategies:
            # --- Plot 1: All Samples (All layers in results) ---
            # Explicitly create figure and axes for "All"
            fig_all, ax_all = plt.subplots(figsize=(12, 8))
            has_data_all = False
            
            color_idx = 0
            
            # --- Plot 2: Correct vs Wrong (All layers in results) ---
            # Only relevant if not metric == 'acc'
            do_split_plot = (metric != "acc")
            if do_split_plot:
                fig_split, ax_split = plt.subplots(figsize=(12, 8))
                has_data_split = False
            
            for model_key, layers_data in results.items():
                for layer, lams_data in layers_data.items():
                    
                    # Store data for sorting: subset -> (xs, ys)
                    plot_dict = {
                        "all": ([], []),
                        "correct": ([], []),
                        "wrong": ([], [])
                    }
                    
                    for lam, stats in lams_data.items():
                        x_val = lam_to_float(lam)
                        
                        # Get Metric Values and Correctness
                        Y_arr = np.array(stats.get("Y", []))
                        if len(Y_arr) == 0: continue
                        
                        if metric == "acc":
                            vals_arr = Y_arr
                        elif metric == "avg_steps":
                            vals_arr = np.array(stats.get(f"steps_count_{strategy}", []))
                        elif metric == "avg_tokens_per_step":
                            vals_arr = np.array(stats.get(f"tokens_per_step_avg_{strategy}", []))
                        elif metric == "avg_total_tokens":
                            vals_arr = np.array(stats.get(f"total_tokens_{strategy}", []))
                        else:
                            vals_arr = np.zeros_like(Y_arr)
                        
                        if len(vals_arr) > 0:
                            all_y_vals.extend(vals_arr.tolist())

                        if len(vals_arr) != len(Y_arr):
                            min_len = min(len(vals_arr), len(Y_arr))
                            vals_arr = vals_arr[:min_len]
                            Y_arr = Y_arr[:min_len]

                        # Calculate Means
                        # All
                        plot_dict["all"][0].append(x_val)
                        plot_dict["all"][1].append(np.mean(vals_arr))
                        
                        # Correct
                        mask_c = (Y_arr == 1)
                        if np.any(mask_c):
                            plot_dict["correct"][0].append(x_val)
                            plot_dict["correct"][1].append(np.mean(vals_arr[mask_c]))
                            
                        # Wrong
                        mask_w = (Y_arr == 0)
                        if np.any(mask_w):
                            plot_dict["wrong"][0].append(x_val)
                            plot_dict["wrong"][1].append(np.mean(vals_arr[mask_w]))

                    # Plotting Lines
                    base_label = f"{model_key}-{layer}"
                    current_color = colors[color_idx % len(colors)]
                    color_idx += 1
                    
                    # 1. All (Solid) -> To Main Figure (ax_all)
                    px, py = plot_dict["all"]
                    if px:
                        sorted_pairs = sorted(zip(px, py), key=lambda x: x[0])
                        sx, sy = zip(*sorted_pairs)
                        ax_all.plot(sx, sy, marker='o', linestyle='-', color=current_color, lw=2, label=f"{base_label}")
                        has_data_all = True
                        
                    # 2. Subsets (only for non-Acc metrics) -> To Split Figure (ax_split)
                    if do_split_plot:
                        # Correct (Dashed)
                        cx, cy = plot_dict["correct"]
                        if cx:
                            sorted_pairs = sorted(zip(cx, cy), key=lambda x: x[0])
                            sx, sy = zip(*sorted_pairs)
                            ax_split.plot(sx, sy, marker='^', linestyle='--', color=current_color, alpha=0.7, label=f"{base_label} (Correct)")
                            has_data_split = True
                            
                        # Wrong (Dotted)
                        wx, wy = plot_dict["wrong"]
                        if wx:
                            sorted_pairs = sorted(zip(wx, wy), key=lambda x: x[0])
                            sx, sy = zip(*sorted_pairs)
                            ax_split.plot(sx, sy, marker='x', linestyle=':', color=current_color, alpha=0.7, label=f"{base_label} (Wrong)")
                            has_data_split = True

            # ylim = robust_ylim(np.array(all_y_vals), low=5, high=95)
            # if ylim is not None:
            #     ax_all.set_ylim(*ylim)
            # Save "All" Plot
            if has_data_all:
                cfg = PLOT_CONFIG.get(metric, {})
                title_suffix = "" if strategy == "main" else f" ({strategy})"
                ax_all.set_title(cfg.get("title", metric) + title_suffix + " (All Samples)")
                ax_all.set_xlabel("Lambda (λ)")
                ax_all.set_ylabel(cfg.get("ylabel", metric))
                ax_all.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax_all.grid(True, alpha=0.3)
                fig_all.tight_layout()
                
                fname_suffix = "" if strategy == "main" else f"_{strategy}"
                out_path = os.path.join(save_dir, f"{filename_prefix}{metric}{fname_suffix}_all.png")
                fig_all.savefig(out_path, dpi=300)
                print(f"Saved plot: {out_path}")
            plt.close(fig_all)
            
            # Save "Split" Plot
            if do_split_plot and has_data_split:
                cfg = PLOT_CONFIG.get(metric, {})
                ax_split.set_title(cfg.get("title", metric) + f" ({strategy}) (Correct vs Wrong)")
                ax_split.set_xlabel("Lambda (λ)")
                ax_split.set_ylabel(cfg.get("ylabel", metric))
                ax_split.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax_split.grid(True, alpha=0.3)
                fig_split.tight_layout()
                
                fname_suffix = f"_{strategy}"
                out_path = os.path.join(save_dir, f"{filename_prefix}{metric}{fname_suffix}_split.png")
                fig_split.savefig(out_path, dpi=300)
                print(f"Saved plot: {out_path}")
                plt.close(fig_split)

def get_top_k_layers(results, k=5):
    """Find the top K layers based on peak accuracy across all lambdas"""
    layer_scores = []

    for model_key, layers_data in results.items():
        for layer, lams_data in layers_data.items():

            best_acc = -1
            best_lam = None

            for lam, stats in lams_data.items():
                Y_arr = stats.get("Y", [])
                if not Y_arr:
                    continue

                acc = np.mean(Y_arr)
                if acc > best_acc:
                    best_acc = acc
                    best_lam = lam

            # record only once per (model, layer)
            layer_scores.append((best_acc, model_key, layer, best_lam))

    # Sort descending by accuracy
    layer_scores.sort(key=lambda x: x[0], reverse=True)

    # Take top K
    top_layers = layer_scores[:k]

    print(f"Top {k} Layers by Peak Accuracy:")
    for score, m, l, lam in top_layers:
        print(f"  {m} - {l} (lam={lam}): {score:.4f}")

    return set((m, l) for _, m, l, _ in top_layers)

def filter_results_by_layers(results, target_layers):
    """Return a new results dict containing only specified layers"""
    new_results = {}
    for model_key, layers_data in results.items():
        for layer, lams_data in layers_data.items():
            if (model_key, layer) in target_layers:
                new_results.setdefault(model_key, {})[layer] = lams_data
    return new_results


# ============================================================
# 5. 主入口 (Main)
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stats Pipeline: Search -> Execute -> Plot")
    parser.add_argument("--base_dir", default=DEFAULT_BASE_DIR, help="Root directory to search")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Where to save JSON and Plots")
    args = parser.parse_args()
    
    # 如果已经有结果，可以直接加载
    json_path = os.path.join(args.out_dir, "statistical_results.json")  
    if os.path.exists(json_path):
        print(f"Loading existing results from {json_path} ...")
        with open(json_path, "r") as f:
            all_results = json.load(f)
    else:
        # 1. 执行扫描与计算
        print("=== Step 1: Searching and Executing Stats Tasks ===")
        all_results = scan_and_process(os.path.join(args.base_dir, TASKS))
    
    # 2. 保存 JSON 结果
    if all_results:
        os.makedirs(args.out_dir, exist_ok=True)
        json_path = os.path.join(args.out_dir, "statistical_results.json")
        print(f"\n=== Step 2: Saving Results to {json_path} ===")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
            
        # 3. Top 5 Analysis (Only plot top 5)
        print(f"\n=== Step 3: Top 5 Layers Analysis ===")
        top_layers_set = get_top_k_layers(all_results, k=5)
        top5_results = filter_results_by_layers(all_results, top_layers_set)
        
        if top5_results:
            top5_dir = os.path.join(args.out_dir, "top5_layers")
            print(f"Generating Top 5 plots in {top5_dir} ...")
            plot_results(top5_results, top5_dir, filename_prefix="top5_")
        
        top_layers_set = get_top_k_layers(all_results, k=10)
        top10_results = filter_results_by_layers(all_results, top_layers_set)
        
        if top10_results:
            top10_dir = os.path.join(args.out_dir, "top10_layers")
            print(f"Generating Top 10 plots in {top10_dir} ...")
            plot_results(top10_results, top10_dir, filename_prefix="top10_")
        print("\nAll Done! 🎉")
    else:
        print("No data found or processed.")