#!/usr/bin/env python3
"""Launch MuSiQue experiments on GPUs 3,4,5"""
import subprocess, os, sys

BASE = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement"
PYTHON = "/common/users/sl2148/anaconda3/envs/fact_yang/bin/python"
OUTBASE = f"{BASE}/exps/steer_runs_e11_e12"
LOGDIR = f"{OUTBASE}/overnight_logs"
os.makedirs(LOGDIR, exist_ok=True)

hf_token = ""
token_path = f"{BASE}/new_exps/.cache/huggingface/token"
if os.path.exists(token_path):
    with open(token_path) as f:
        hf_token = f.read().strip()

MODEL = "Qwen/Qwen2.5-3B-Instruct"
GPT_VEC = f"{BASE}/exps/gpt_rewrites_unified_new/Qwen_Qwen2.5-3B-Instruct/vectors_50_old/Qwen_Qwen2.5-3B-Instruct_applied/steering_vector.pt"
LM_VEC = f"{BASE}/exps/large_model_rewrites_unified_new/Qwen_Qwen2.5-3B-Instruct/vectors_50_paired_Qwen_Qwen2.5-7B-Instruct/Qwen_Qwen2.5-3B-Instruct_applied/steering_vector.pt"

CONFIGS = [
    {"gpu": 3, "mode": "BASELINE",    "layer": 0, "lam": 0.0,  "vec": GPT_VEC},
    {"gpu": 4, "mode": "GPT_REWRITE", "layer": 6, "lam": 4.0,  "vec": GPT_VEC},
    {"gpu": 5, "mode": "LARGE_MODEL", "layer": 6, "lam": 0.45, "vec": LM_VEC},
]

TASK = "longbench_musique"
procs = []

for cfg in CONFIGS:
    safe_model = MODEL.replace("/", "_")
    lam_tag = f"{cfg['lam']:.6f}".rstrip("0").rstrip(".")
    outdir = f"{OUTBASE}/{cfg['mode']}/{safe_model}/{TASK}/L{cfg['layer']}/lam_{lam_tag}"
    logfile = f"{LOGDIR}/{cfg['mode']}_{TASK}_L{cfg['layer']}_lam{lam_tag}.log"
    
    existing = [f for f in os.listdir(outdir) if f.startswith("results_") and f.endswith(".json")] if os.path.isdir(outdir) else []
    if existing:
        print(f"[SKIP] {cfg['mode']} {TASK} - already done")
        continue
    
    os.makedirs(outdir, exist_ok=True)
    
    cmd = [
        PYTHON, "-m", "lm_eval",
        "--model", "steer_hf",
        "--model_args", f"pretrained={MODEL},dtype=float16,steer_layer={cfg['layer']},steer_lambda={cfg['lam']},steer_vec_path={cfg['vec']}",
        "--tasks", TASK,
        "--device", "cuda:0",
        "--num_fewshot", "0",
        "--batch_size", "1",
        "--gen_kwargs", "do_sample=false,temperature=0,max_gen_toks=512",
        "--output_path", outdir,
        "--log_samples",
        "--trust_remote_code",
        "--apply_chat_template",
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"])
    env["HF_TOKEN"] = hf_token
    env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    
    log_fh = open(logfile, "w")
    p = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env, cwd=BASE)
    procs.append((cfg["mode"], p, log_fh))
    print(f"[START] GPU={cfg['gpu']} {cfg['mode']} {TASK} PID={p.pid}")

print(f"\nWaiting for {len(procs)} MuSiQue tasks...")
for mode, p, fh in procs:
    rc = p.wait()
    fh.close()
    status = "DONE" if rc == 0 else "FAIL"
    print(f"[{status}] {mode} {TASK} (rc={rc})")

print("\nAll MuSiQue tasks finished.")
