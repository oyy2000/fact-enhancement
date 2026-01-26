# 03_pipeline_runner.py
import os
import glob
import json
import multiprocessing as mp
import subprocess
import importlib.util

# 1. 动态加载绘图库
spec = importlib.util.spec_from_file_location("plots", "05_plots_concise.py")
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)

# 2. 配置
BASE_DIR = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/gpt_rewrites_unified/Qwen_Qwen2.5-3B-Instruct/vectors_50_expert_leap/Qwen_Qwen2.5-3B-Instruct_applied/gsm8k_cot_zeroshot_unified"
STATS_OUT_DIR = "prm_results/stats_only"
MERGED_FILE = f"{STATS_OUT_DIR}/results_merged.json"

# 3. 任务扫描逻辑 (简化版)
def get_jobs():
    jobs = []
    # 这里填入你的文件扫描逻辑
    # 示例: 遍历 BASE_DIR 找到 layer, lam 和 jsonl
    # ...
    # jobs.append({"model": "Qwen...", "layer": "L10", "lam": "lam10", "jsonl": "path...", "gen_model": "Qwen/..."})
    return jobs

def worker_stats(job):
    out_file = f"{STATS_OUT_DIR}/chunk_{job['layer']}_{job['lam']}.json"
    cmd = [
        "python", "06_calc_stats.py",
        "--model_name", job['model'],
        "--gen_model", job['gen_model'],
        "--layer", job['layer'],
        "--lam", job['lam'],
        "--jsonl", job['jsonl'],
        "--out", out_file
    ]
    subprocess.run(cmd)

def merge_results(out_dir):
    final = {}
    for f in glob.glob(f"{out_dir}/chunk_*.json"):
        data = json.load(open(f))
        # Deep Merge
        for m in data:
            final.setdefault(m, {})
            for l in data[m]:
                final[m].setdefault(l, {})
                final[m][l].update(data[m][l])
    
    with open(MERGED_FILE, "w") as f:
        json.dump(final, f, indent=2)
    return final

if __name__ == "__main__":
    # A. 运行统计
    jobs = get_jobs()
    pool = mp.Pool(16) # CPU 任务，可以并行更多
    pool.map(worker_stats, jobs)
    pool.close()
    pool.join()
    
    # B. 合并结果
    print("Merging stats...")
    merged_data = merge_results(STATS_OUT_DIR)
    
    # C. 画图
    print("Plotting...")
    plots.setup_plotting(merged_data, STATS_OUT_DIR)
    plots.plot_all() # 这里会生成你需要的4张 Lambda 趋势图
    
    print(f"Done! Check plots in {STATS_OUT_DIR}/all/")