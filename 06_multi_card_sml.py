# run_master.py
import os, json, glob, multiprocessing as mp

# ===== configs ==================================================

FOLDER = "./prm_out_qwen_family"   # ⭐ 你想要的 folder
os.makedirs(FOLDER, exist_ok=True)


NUM_GPUS = 4
GPU_IDS = [0,2,3, 4]# list(range(NUM_GPUS))

BASE_DIR = "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_qwen_familiy/gsm8k_cot_zeroshot"
L_values = ["L8"]
lam_values = ["BASELINE"]   # 横轴维度

model_names = {
    # "Qwen2.5-0.5B-Instruct":  "Qwen__Qwen2.5-0.5B-Instruct",
    # "Qwen2.5-1.5B-Instruct":  "Qwen__Qwen2.5-1.5B-Instruct",
    # "Qwen2.5-3B-Instruct":  "Qwen__Qwen2.5-3B-Instruct",
    # "Qwen2.5-7B-Instruct":  "Qwen__Qwen2.5-7B-Instruct",
    "Qwen2.5-14B-Instruct":  "Qwen__Qwen2.5-14B-Instruct",
}

# ===== prepare list of jobs ======================================

jobs = []

for model_name, model_folder in model_names.items():
    for L in L_values:
        for lam in lam_values:
            folder = f"{BASE_DIR}/{model_name}_{L}_{lam}/{model_folder}/"
            pattern = os.path.join(folder, "samples_gsm8k_cot_zeroshot_*.jsonl")

            files = sorted(glob.glob(pattern))
            if len(files)==0:
                print(f"⚠ Missing: {model_name} {L} {lam}")
                continue

            jsonl_file = files[-1]

            jobs.append((model_name, L, lam, jsonl_file))

print(f"Total jobs = {len(jobs)}")


# ===== worker process ============================================
def worker(job_idx, job, gpu_id):
    model_name, L, lam, jsonl = job
    out_file = f"{FOLDER}/results_chunk_{job_idx}.json"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = (
        f"python 07_run_prm_single.py "
        f"--model '{model_name}' "
        f"--layer {L} "
        f"--lam {lam} "
        f"--jsonl {jsonl} "
        f"--out {out_file}"
    )
    print(f"[GPU{gpu_id}] → {cmd}")
    os.system(cmd)


# ===== launch pool ===============================================

if __name__ == "__main__":
    pool = mp.Pool(NUM_GPUS)

    for idx, job in enumerate(jobs):
        gpu_id = GPU_IDS[idx % NUM_GPUS]
        pool.apply_async(worker, args=(idx, job, gpu_id))

    pool.close()
    pool.join()

    print("✔ All GPU jobs finished.")

    # ===== merge chunks ===========================================
    final = {}

    for f in sorted(glob.glob("results_chunk_*.json")):
        part = json.load(open(f))

        for model in part:
            final.setdefault(model, {})
            for L in part[model]:
                final[model].setdefault(L, {})
                for lam in part[model][L]:
                    final[model][L][lam] = part[model][L][lam]

    with open("results_merged.json", "w") as f:
        json.dump(final, f, indent=2)

    print("🎉 Final merged JSON → results_merged.json")
