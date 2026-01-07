import os
import json
import glob
import multiprocessing as mp

# ============================================================
# CONFIGS
# ============================================================

FOLDER = "./prm_out_llama_family"
os.makedirs(FOLDER, exist_ok=True)

NUM_GPUS = 8
GPU_IDS = [0,1,2,3,4,5,6,7]

BASE_DIR = "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_llama_family/gsm8k_cot_zeroshot"

lam_values = [
    # "lam-0p5","lam-1p0","lam-1p5","lam-2p0",
    "BASELINE",
    # "lam0p5","lam1p0","lam1p5","lam2p0"
]

# ============================================================
# MODEL DEFINITIONS
# ============================================================

# logical model name -> HF repo id
MODEL_MAP = {
    # "Qwen2.5-7B-Instruct":   "Qwen/Qwen2.5-7B-Instruct",
    # "Qwen2.5-3B-Instruct":   "Qwen/Qwen2.5-3B-Instruct",
    # "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    # "Qwen2.5-14B-Instruct":  "Qwen/Qwen2.5-14B-Instruct",
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
}

# logical model name -> layer indices
MODEL_TO_LAYERS = {
    # "Qwen2.5-7B-Instruct":   [14, 24, 28],
    # "Qwen2.5-1.5B-Instruct": [14, 24, 28],
    # "Qwen2.5-3B-Instruct":   [18, 32, 36],/
    "Llama-3.1-8B-Instruct": [8],
    "Llama-3.2-1B-Instruct": [8],
    "Llama-3.2-3B-Instruct": [8]
    # "Qwen2.5-14B-Instruct":  [16, 24, 32],
}

# folder names produced by lm-eval
MODEL_FOLDER_MAP = {
    k: v.replace("/", "__")
    for k, v in MODEL_MAP.items()
}


# ============================================================
# PREPARE JOB LIST
# ============================================================

jobs = []

for model_name, gen_model_name in MODEL_MAP.items():

    if model_name not in MODEL_TO_LAYERS:
        print(f"⚠ No layer config for model: {model_name}")
        continue

    model_folder = MODEL_FOLDER_MAP[model_name]
    layer_indices = MODEL_TO_LAYERS[model_name]

    for layer_idx in layer_indices:
        L = f"L{layer_idx}"

        for lam in lam_values:
            folder = f"{BASE_DIR}/{model_name}_{L}_{lam}/{model_folder}/"
            print(folder)
            pattern = os.path.join(folder, "samples_gsm8k_cot_zeroshot_*.jsonl")

            files = sorted(glob.glob(pattern))
            if len(files) == 0:
                print(f"⚠ Missing: {model_name} {L} {lam}")
                continue

            jsonl_file = files[-1]
            jobs.append((model_name, gen_model_name, L, lam, jsonl_file))

print(f"Total jobs = {len(jobs)}")

# ============================================================
# WORKER
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
        f"--out {out_file}"
    )

    print(f"[GPU{gpu_id}] → {cmd}")
    os.system(cmd)

# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    pool = mp.Pool(NUM_GPUS)

    for idx, job in enumerate(jobs):
        gpu_id = GPU_IDS[idx % NUM_GPUS]
        pool.apply_async(worker, args=(idx, job, gpu_id))

    pool.close()
    pool.join()

    print("✔ All GPU jobs finished.")

    # ============================================================
    # MERGE RESULTS
    # ============================================================

    final = {}

    for f in sorted(glob.glob(f"{FOLDER}/results_chunk_*.json")):
        part = json.load(open(f))

        for model in part:
            final.setdefault(model, {})
            for L in part[model]:
                final[model].setdefault(L, {})
                for lam in part[model][L]:
                    final[model][L][lam] = part[model][L][lam]

    merged_path = f"{FOLDER}/results_merged.json"
    with open(merged_path, "w") as f:
        json.dump(final, f, indent=2)

    print(f"🎉 Final merged JSON → {merged_path}")
