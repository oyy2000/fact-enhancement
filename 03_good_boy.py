#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import scan_and_process

# ============================================================
# 1. 配置区域 (Configuration)
# ============================================================

STATUS = "gpt_rewrite"  # or "large_model_rewrite"
PROMPT_STYLE = "old"
TASKS = "gsm8k_cot_zeroshot_unified_selected_layers"

REWRITE_MODEL = "Qwen/Qwen2.5-3B-Instruct".replace("/", "_")
TARGET_MODEL = "Qwen/Qwen2.5-3B-Instruct".replace("/", "_")

if STATUS == "large_model_rewrite":
    FOLDER_NAME = "large_model_rewrites_unified_new"
    VECTOR_PATH = f"vectors_50_paired_{REWRITE_MODEL}"
elif STATUS == "gpt_rewrite":
    FOLDER_NAME = "gpt_rewrites_unified_new"
    VECTOR_PATH = f"vectors_50_{PROMPT_STYLE}"

DEFAULT_BASE_DIR = f"./{FOLDER_NAME}/{TARGET_MODEL}/{VECTOR_PATH}/{TARGET_MODEL}_applied/"
DEFAULT_OUT_DIR = os.path.join(DEFAULT_BASE_DIR, TASKS, "aaresults/stats_output")


# ============================================================
# 2. 工具函数
# ============================================================

def lam_to_float(lam):
    """将 lam1, lam0p5, BASELINE 转为 float"""
    if isinstance(lam, (int, float)):
        return float(lam)
    if isinstance(lam, str) and "BASELINE" in lam.upper():
        return 0.0
    s = str(lam).lower().replace("lam", "").replace("p", ".")
    try:
        return float(s)
    except Exception:
        return -1.0


def get_top_k_layers(results, k=10):
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
                acc = float(np.mean(np.array(Y_arr)))
                if acc > best_acc:
                    best_acc = acc
                    best_lam = lam

            layer_scores.append((best_acc, model_key, layer, best_lam))

    layer_scores.sort(key=lambda x: x[0], reverse=True)
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
# 3. 新需求：Top10 三个指标竖排 + 共用图例 + 去标题 + 高清 + 大字体
# ============================================================

def plot_top10_double_all_vertical(
    results_top10,
    save_dir,
    filename="top10_double_all_vertical.png",
    dpi=600,
    figsize=(7, 12),
    base_fontsize=18,
    tick_fontsize=14,
    lw=2.5,
    marker_size=6,
):
    """
    画三张竖排子图（共用 x 轴），分别是：
      - avg_steps (double) all
      - avg_tokens_per_step (double) all
      - avg_total_tokens (double) all

    需求：
      - 增大分辨率/字体
      - 共用图例
      - 去掉标题
    """
    if not results_top10:
        print("No top10 results to plot.")
        return

    os.makedirs(save_dir, exist_ok=True)

    # 全局字体
    plt.rcParams.update({
        "font.size": base_fontsize,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": tick_fontsize,
        "ytick.labelsize": tick_fontsize,
        "legend.fontsize": tick_fontsize,
    })

    metric_specs = [
        ("avg_steps", "Average Steps", "steps_count_double"),
        ("avg_tokens_per_step", "Tokens / Step", "tokens_per_step_avg_double"),
        ("avg_total_tokens", "Total Tokens", "total_tokens_double"),
    ]

    # 子图：竖排三行，共用 x
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=figsize,
        sharex=True,
        constrained_layout=False,  # 我们后面用 tight_layout + legend 来精调
    )

    colors = plt.cm.tab10.colors
    color_idx = 0

    # 为了共用图例：只在第一次出现时收集 handle/label
    legend_handles = []
    legend_labels = []
    seen_labels = set()

    for model_key, layers_data in results_top10.items():
        for layer, lams_data in layers_data.items():
            base_label = f"{model_key}-{layer}"
            current_color = colors[color_idx % len(colors)]
            color_idx += 1

            # 收集三个指标的数据：每个指标一条线（all-samples mean）
            # {metric_key: (xs, ys)}
            line_data = {ms[0]: ([], []) for ms in metric_specs}

            for lam, stats in lams_data.items():
                x_val = lam_to_float(lam)

                Y_arr = np.array(stats.get("Y", []))
                if Y_arr.size == 0:
                    continue

                for metric_key, _, field in metric_specs:
                    vals = np.array(stats.get(field, []))
                    if vals.size == 0:
                        continue

                    # 对齐长度（保险）
                    if vals.size != Y_arr.size:
                        m = min(vals.size, Y_arr.size)
                        vals = vals[:m]

                    line_data[metric_key][0].append(x_val)
                    line_data[metric_key][1].append(float(np.mean(vals)))

            # 画三行
            for ax, (metric_key, ylabel, _) in zip(axes, metric_specs):
                xs, ys = line_data[metric_key]
                if not xs:
                    continue

                sorted_pairs = sorted(zip(xs, ys), key=lambda t: t[0])
                sx, sy = zip(*sorted_pairs)

                (line_handle,) = ax.plot(
                    sx,
                    sy,
                    marker="o",
                    markersize=marker_size,
                    linestyle="-",
                    linewidth=lw,
                    color=current_color,
                    label=base_label,
                )

                # legend 收集（只收一次）
                if base_label not in seen_labels:
                    legend_handles.append(line_handle)
                    legend_labels.append(base_label)
                    seen_labels.add(base_label)

    # 轴样式：无标题，只有 ylabel；x 轴 label 放最下面
    for ax, (_, ylabel, _) in zip(axes, metric_specs):
        ax.set_title("")  # 去掉标题
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Lambda (λ)")

    # 共用图例：放在整张图右侧
    # 预留右边空间，避免 legend 挤压子图
    # fig.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    # 给顶部留空间（legend 在最上面）
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,              # 👈 可以根据 layer 数调，比如 3 / 4 / 5
        frameon=False,
    )

    # fig.legend(
    #     legend_handles,
    #     legend_labels,
    #     loc="center left",
    #     bbox_to_anchor=(0.80, 0.5),
    #     frameon=False,
    # )

    out_path = os.path.join(save_dir, filename)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined vertical plot: {out_path}")


# ============================================================
# 4. 主入口 (Main)
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stats Pipeline + Top10 Double-All Vertical Plot")
    parser.add_argument("--base_dir", default=DEFAULT_BASE_DIR, help="Root directory to search")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Where to save JSON and Plots")
    parser.add_argument("--k", type=int, default=10, help="Top-K layers to keep (default: 10)")
    args = parser.parse_args()

    json_path = os.path.join(args.out_dir, "statistical_results.json")
    if os.path.exists(json_path):
        print(f"Loading existing results from {json_path} ...")
        with open(json_path, "r") as f:
            all_results = json.load(f)
    else:
        print("=== Step 1: Searching and Executing Stats Tasks ===")
        all_results = scan_and_process(os.path.join(args.base_dir, TASKS))

        if all_results:
            os.makedirs(args.out_dir, exist_ok=True)
            print(f"\n=== Step 2: Saving Results to {json_path} ===")
            with open(json_path, "w") as f:
                json.dump(all_results, f, indent=2)

    if not all_results:
        print("No data found or processed.")
        raise SystemExit(0)

    print(f"\n=== Step 3: Top-{args.k} Layers Analysis ===")
    top_layers_set = get_top_k_layers(all_results, k=args.k)
    topk_results = filter_results_by_layers(all_results, top_layers_set)

    if not topk_results:
        print("Top-k filtering produced empty results.")
        raise SystemExit(0)

    # 输出目录：top10_layers（保持你原来的习惯）
    topk_dir = os.path.join(args.out_dir, f"top{args.k}_layers")
    os.makedirs(topk_dir, exist_ok=True)

    # 生成你要的：三张竖着拼在一张图里（double + all）
    plot_top10_double_all_vertical(
        results_top10=topk_results,
        save_dir=topk_dir,
        filename=f"top{args.k}_avg_steps_tokens_total_double_all_vertical.png",
        dpi=400,              # 分辨率更高
        figsize=(14, 28),     # 图更大
        base_fontsize=30,     # 字体更大
        tick_fontsize=30,
        lw=3.0,
        marker_size=7,
    )

    print("\nAll Done! 🎉")
