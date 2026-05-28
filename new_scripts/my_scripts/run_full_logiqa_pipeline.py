#!/usr/bin/env python3
"""
Full end-to-end pipeline:
  0a) Generate LogiQA CoT with Qwen2.5-3B
  0b) Generate LogiQA CoT with Qwen2.5-7B
  1a) GPT-5.1 dense rewrite of 3B outputs
  1b) Pair 3B/7B outputs for InFamilySteer
  2)  Extract steering vectors (DenseSteer + InFamily)
  3)  Lambda sweep on LogiQA eval task (lm_eval `logiqa`)
  4)  Report best params

Run:
  CUDA_VISIBLE_DEVICES=0 python new_scripts/my_scripts/run_full_logiqa_pipeline.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
EXPS = ROOT / "exps" / "logiqa_densesteer"

os.chdir(ROOT)

# Load API credentials
env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()


def run(cmd: list[str], desc: str, env_extra: dict | None = None, timeout: int = 14400):
    print(f"\n{'='*70}")
    print(f"  {desc}")
    print(f"  CMD: {' '.join(cmd[:8])}{'...' if len(cmd) > 8 else ''}")
    print(f"{'='*70}", flush=True)
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, cwd=str(ROOT), timeout=timeout)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"  [FAIL] returncode={proc.returncode} after {elapsed:.0f}s", flush=True)
        sys.exit(proc.returncode)
    print(f"  [OK] {elapsed:.0f}s", flush=True)


def run_lm_eval(task, steer_vec, layer, lam, outdir, gpu=None, batch_size=16, limit=None):
    model_args = (
        f"pretrained=Qwen/Qwen2.5-3B-Instruct,dtype=float16,"
        f"steer_layer={layer},steer_lambda={lam},steer_vec_path={steer_vec}"
    )
    cmd = [
        PY, "-m", "lm_eval",
        "--model", "steer_hf",
        "--model_args", model_args,
        "--tasks", task,
        "--device", "cuda:0",
        "--num_fewshot", "0",
        "--batch_size", str(batch_size),
        "--gen_kwargs", "do_sample=false,temperature=0,max_gen_toks=512",
        "--output_path", str(outdir),
        "--log_samples",
        "--trust_remote_code",
        "--apply_chat_template",
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    hf_token_path = ROOT / "new_exps" / ".cache" / "huggingface" / "token"
    env_extra = {"CUDA_VISIBLE_DEVICES": str(gpu or GPU)}
    if hf_token_path.is_file():
        tok = hf_token_path.read_text().strip()
        env_extra["HF_TOKEN"] = tok
        env_extra["HUGGING_FACE_HUB_TOKEN"] = tok
    lam_tag = f"{lam:.6f}".rstrip("0").rstrip(".")
    run(cmd, f"lm_eval {task} L{layer} λ={lam}", env_extra=env_extra)


def extract_score(results_dir, task):
    for f in sorted(Path(results_dir).rglob("results_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        r = data.get("results", {})
        if task in r:
            for mk, mv in r[task].items():
                if "stderr" not in mk and mk != "alias":
                    return mk, mv
    return None, None


# ========================= PATHS =========================
SAMPLE_DIR = EXPS / "samples"
PAIR_DIR = EXPS / "paired"
GPT_DIR = EXPS / "gpt_rewrites" / "Qwen_Qwen2.5-3B-Instruct"
VEC_DIR = EXPS / "vectors" / "Qwen_Qwen2.5-3B-Instruct"
SWEEP_DIR = EXPS / "sweep"

JSONL_3B = SAMPLE_DIR / "Qwen_Qwen2.5-3B-Instruct_logiqa_cot_train.jsonl"
JSONL_7B = SAMPLE_DIR / "Qwen_Qwen2.5-7B-Instruct_logiqa_cot_train.jsonl"
PAIR_JSON = PAIR_DIR / "Qwen3B_paired_Qwen7B_logiqa.json"
GPT_JSON = GPT_DIR / "rewritten_old.json"

DENSE_VEC_DIR = VEC_DIR / "N50_dense_gpt_old" / "Qwen_Qwen2.5-3B-Instruct_applied"
INFAMILY_VEC_DIR = VEC_DIR / "N50_infamily_7b" / "Qwen_Qwen2.5-3B-Instruct_applied"

LIMIT_GEN = 800
GPU = os.environ.get("PIPELINE_GPU", "0")
GPU_7B = os.environ.get("PIPELINE_GPU_7B", "0,1")

for d in [SAMPLE_DIR, PAIR_DIR, GPT_DIR, VEC_DIR, SWEEP_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ========================= STEP 0a: 3B CoT =========================
if not JSONL_3B.exists() or JSONL_3B.stat().st_size < 100:
    run(
        [PY, "new_scripts/my_scripts/logiqa_generate_cot.py",
         "--model", "Qwen/Qwen2.5-3B-Instruct",
         "--split", "train", "--limit", str(LIMIT_GEN),
         "--out_jsonl", str(JSONL_3B)],
        "Step 0a: Generate LogiQA CoT with Qwen2.5-3B",
        env_extra={"CUDA_VISIBLE_DEVICES": GPU},
    )
else:
    print(f"\n[SKIP] 0a: {JSONL_3B} already exists")


# ========================= STEP 0b: 7B CoT =========================
if not JSONL_7B.exists() or JSONL_7B.stat().st_size < 100:
    run(
        [PY, "new_scripts/my_scripts/logiqa_generate_cot.py",
         "--model", "Qwen/Qwen2.5-7B-Instruct",
         "--split", "train", "--limit", str(LIMIT_GEN),
         "--out_jsonl", str(JSONL_7B)],
        "Step 0b: Generate LogiQA CoT with Qwen2.5-7B",
        env_extra={"CUDA_VISIBLE_DEVICES": GPU_7B},
    )
else:
    print(f"\n[SKIP] 0b: {JSONL_7B} already exists")


# ========================= STEP 1a: GPT-5.1 Dense Rewrite =========================
if not GPT_JSON.exists() or GPT_JSON.stat().st_size < 100:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("BASE_URL", "")
    if not api_key:
        print("\n[ERROR] No OPENAI_API_KEY — cannot do GPT rewrite. Set it in .env")
        sys.exit(1)
    run(
        [PY, "00_gpt_modification.py",
         "--in_jsonl", str(JSONL_3B),
         "--out_json", str(GPT_JSON),
         "--prompt_style", "old",
         "--rewrite_last_n", "100000"],
        "Step 1a: GPT-5.1 dense rewrite of 3B LogiQA outputs",
        env_extra={"OPENAI_API_KEY": api_key, "BASE_URL": base_url} if base_url else {"OPENAI_API_KEY": api_key},
        timeout=7200,
    )
else:
    print(f"\n[SKIP] 1a: {GPT_JSON} already exists")


# ========================= STEP 1b: Pair 3B/7B =========================
if not PAIR_JSON.exists() or PAIR_JSON.stat().st_size < 100:
    run(
        [PY, "new_scripts/my_scripts/logiqa_pair_infamily.py",
         "--small_jsonl", str(JSONL_3B),
         "--large_jsonl", str(JSONL_7B),
         "--out_json", str(PAIR_JSON),
         "--small_model", "Qwen/Qwen2.5-3B-Instruct",
         "--large_model", "Qwen/Qwen2.5-7B-Instruct"],
        "Step 1b: Pair 3B/7B for InFamilySteer",
    )
else:
    print(f"\n[SKIP] 1b: {PAIR_JSON} already exists")


# ========================= STEP 2a: Extract DenseSteer Vector =========================
dense_vec = DENSE_VEC_DIR / "steering_vector.pt"
if not dense_vec.exists():
    run(
        [PY, "new_scripts/my_scripts/logiqa_extract_steering.py",
         "--model", "Qwen/Qwen2.5-3B-Instruct",
         "--in_path", str(GPT_JSON),
         "--num_examples", "50",
         "--layers", "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18",
         "--tag", "dense_gpt_old",
         "--domain", "logiqa"],
        "Step 2a: Extract DenseSteer (LogiQA GPT rewrite) vector",
        env_extra={"CUDA_VISIBLE_DEVICES": GPU},
    )
else:
    print(f"\n[SKIP] 2a: {dense_vec} already exists")


# ========================= STEP 2b: Extract InFamily Vector =========================
infam_vec = INFAMILY_VEC_DIR / "steering_vector.pt"
if not infam_vec.exists():
    run(
        [PY, "new_scripts/my_scripts/logiqa_extract_steering.py",
         "--model", "Qwen/Qwen2.5-3B-Instruct",
         "--in_path", str(PAIR_JSON),
         "--num_examples", "50",
         "--layers", "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18",
         "--tag", "infamily_7b",
         "--domain", "logiqa"],
        "Step 2b: Extract InFamilySteer (LogiQA 3B/7B) vector",
        env_extra={"CUDA_VISIBLE_DEVICES": GPU},
    )
else:
    print(f"\n[SKIP] 2b: {infam_vec} already exists")


# ========================= STEP 3: Lambda × Layer Sweep =========================
print("\n" + "="*70)
print("  Step 3: Lambda × Layer sweep on logiqa (lm_eval)")
print("="*70, flush=True)

SWEEP_LAYERS = [2, 4, 6, 8, 10, 12, 14, 16, 18]
SWEEP_LAMBDAS_DENSE = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0, 4.0]
SWEEP_LAMBDAS_INFAM = [-1.0, -0.5, -0.3, 0.3, 0.45, 0.5, 0.7, 1.0]

sweep_results = []

for mode, vec_dir, lambdas in [
    ("DenseSteer", DENSE_VEC_DIR, SWEEP_LAMBDAS_DENSE),
    ("InFamily", INFAMILY_VEC_DIR, SWEEP_LAMBDAS_INFAM),
]:
    vec_path = vec_dir / "steering_vector.pt"
    if not vec_path.exists():
        print(f"  [SKIP] {mode}: vector not found at {vec_path}")
        continue

    for layer in SWEEP_LAYERS:
        for lam in lambdas:
            lam_tag = f"{lam:.6f}".rstrip("0").rstrip(".")
            outdir = SWEEP_DIR / mode / f"L{layer}" / f"lam_{lam_tag}"
            
            existing = list(outdir.rglob("results_*.json"))
            if existing:
                mk, score = extract_score(outdir, "logiqa")
                if score is not None:
                    sweep_results.append({"mode": mode, "layer": layer, "lam": lam, "metric": mk, "score": score})
                    print(f"  [CACHED] {mode} L{layer} λ={lam}: {mk}={score:.4f}")
                    continue

            try:
                run_lm_eval("logiqa", str(vec_path), layer, lam, str(outdir), batch_size=16)
                mk, score = extract_score(outdir, "logiqa")
                if score is not None:
                    sweep_results.append({"mode": mode, "layer": layer, "lam": lam, "metric": mk, "score": score})
                    print(f"  [RESULT] {mode} L{layer} λ={lam}: {mk}={score:.4f}")
            except SystemExit:
                print(f"  [FAIL] {mode} L{layer} λ={lam} — skipping")
                continue

# Baseline (no steering)
baseline_dir = SWEEP_DIR / "BASELINE" / "L0" / "lam_0"
baseline_existing = list(baseline_dir.rglob("results_*.json"))
if not baseline_existing:
    gsm8k_vec = (
        ROOT / "exps" / "gpt_rewrites_unified_new"
        / "Qwen_Qwen2.5-3B-Instruct" / "vectors_50_old"
        / "Qwen_Qwen2.5-3B-Instruct_applied" / "steering_vector.pt"
    )
    run_lm_eval("logiqa", str(gsm8k_vec), 0, 0.0, str(baseline_dir), batch_size=16)

mk_base, score_base = extract_score(baseline_dir, "logiqa")
if score_base is not None:
    sweep_results.append({"mode": "Baseline", "layer": 0, "lam": 0.0, "metric": mk_base, "score": score_base})


# ========================= STEP 4: Report =========================
print("\n" + "="*70)
print("  SWEEP RESULTS")
print("="*70)

if score_base is not None:
    print(f"\n  Baseline: {mk_base} = {score_base:.4f}")

for mode in ["DenseSteer", "InFamily"]:
    mode_results = [r for r in sweep_results if r["mode"] == mode]
    if not mode_results:
        continue
    best = max(mode_results, key=lambda x: x["score"])
    print(f"\n  {mode} — Best: L{best['layer']} λ={best['lam']}  {best['metric']}={best['score']:.4f}")
    if score_base is not None:
        delta = (best["score"] - score_base) * 100
        print(f"            Δ vs baseline: {'+' if delta >= 0 else ''}{delta:.2f} pp")

    print(f"\n  {mode} all results (sorted by score):")
    for r in sorted(mode_results, key=lambda x: -x["score"])[:15]:
        d = (r["score"] - (score_base or 0)) * 100
        print(f"    L{r['layer']:>2} λ={r['lam']:>6}  {r['score']:.4f}  ({'+' if d >= 0 else ''}{d:.2f})")

# Save summary
summary_path = EXPS / "sweep_summary.json"
with open(summary_path, "w") as f:
    json.dump({"baseline_score": score_base, "results": sweep_results}, f, indent=2)
print(f"\n  Summary saved: {summary_path}")

print("\n" + "="*70)
print("  PIPELINE COMPLETE")
print("="*70)
