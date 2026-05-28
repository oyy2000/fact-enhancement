#!/usr/bin/env python3
"""
Lambda sweep for E4 steering vector on GSM8K (100 samples).
Sweeps lambda from -1.0 to 1.0 in 0.1 steps, layer 6.
Uses 8 GPUs in parallel (up to 8 concurrent jobs).
"""
import subprocess
import os
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement"
MODEL = "Qwen/Qwen2.5-3B-Instruct"
VEC_PATH = os.path.join(BASE, "control_experiments", "Qwen_Qwen2.5-3B-Instruct",
                         "vectors_e4", "steering_vector.pt")
LAYER = 6
LIMIT = 100
NUM_GPUS = 8

lambdas = [round(-1.0 + i * 0.1, 1) for i in range(21)]  # -1.0 to 1.0


def run_one(lam, gpu):
    tag = f"sweep_e4_L{LAYER}_lam{lam}"
    out_dir = os.path.join(BASE, "control_experiments", "Qwen_Qwen2.5-3B-Instruct", tag)

    cmd = [
        "lm_eval",
        "--model", "steer_hf",
        "--model_args", f"pretrained={MODEL},dtype=float16,"
                        f"steer_layer={LAYER},steer_lambda={lam},"
                        f"steer_vec_path={VEC_PATH},"
                        "trust_remote_code=True",
        "--tasks", "gsm8k_cot_zeroshot_unified",
        "--batch_size", "64",
        "--num_fewshot", "0",
        "--limit", str(LIMIT),
        "--output_path", out_dir,
        "--log_samples",
        "--trust_remote_code",
        "--gen_kwargs", "do_sample=False,temperature=0,max_gen_toks=2048",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    # Parse results
    try:
        res_dir = os.path.join(out_dir, "Qwen__Qwen2.5-3B-Instruct")
        res_files = [f for f in os.listdir(res_dir) if f.startswith("results_")]
        if res_files:
            with open(os.path.join(res_dir, sorted(res_files)[-1])) as f:
                res = json.load(f)
            metrics = res["results"]["gsm8k_cot_zeroshot_unified"]
            strict = metrics.get("exact_match,strict-match", 0)
            flexible = metrics.get("exact_match,flexible-extract", 0)
            return lam, {"strict": strict, "flexible": flexible}
    except Exception as e:
        return lam, {"strict": 0, "flexible": 0, "error": str(e)}

    return lam, {"strict": 0, "flexible": 0, "error": "no results file"}


def main():
    print(f"Sweeping {len(lambdas)} lambdas on {NUM_GPUS} GPUs (limit={LIMIT})")
    print(f"Lambdas: {lambdas}")
    sys.stdout.flush()

    results = {}

    with ProcessPoolExecutor(max_workers=NUM_GPUS) as pool:
        futures = {}
        for i, lam in enumerate(lambdas):
            gpu = i % NUM_GPUS
            fut = pool.submit(run_one, lam, gpu)
            futures[fut] = lam

        for fut in as_completed(futures):
            lam, res = fut.result()
            results[lam] = res
            s, f = res["strict"], res["flexible"]
            print(f"  lam={lam:>5.1f}  strict={s:.4f}  flexible={f:.4f}")
            sys.stdout.flush()

    # Summary sorted by lambda
    print(f"\n{'='*60}")
    print(f"  SWEEP SUMMARY (Layer {LAYER}, {LIMIT} samples, {NUM_GPUS} GPUs)")
    print(f"{'='*60}")
    print(f"{'Lambda':>8}  {'Strict':>8}  {'Flexible':>8}")
    print(f"{'-'*8}  {'-'*8}  {'-'*8}")

    best_lam = None
    best_flex = -1
    for lam in lambdas:
        if lam not in results:
            continue
        s = results[lam]["strict"]
        f = results[lam]["flexible"]
        if f > best_flex:
            best_flex = f
            best_lam = lam
        print(f"{lam:>8.1f}  {s:>8.4f}  {f:>8.4f}")

    print(f"\nBest lambda = {best_lam} (flexible = {best_flex:.4f})")

    # Save summary
    summary_path = os.path.join(BASE, "control_experiments", "Qwen_Qwen2.5-3B-Instruct",
                                  "sweep_e4_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"layer": LAYER, "limit": LIMIT, "num_gpus": NUM_GPUS,
                   "results": {str(k): v for k, v in results.items()},
                   "best_lambda": best_lam, "best_flexible": best_flex}, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
