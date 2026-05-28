#!/usr/bin/env python3
"""
E8 Calibration Size Sweep: For each calibration size, sweep layers x lambdas.
Lambda ranges are computed from per-layer vector norms:
    norm <  10  -> step = 0.5, range [-5,  5]
    norm 10~100 -> step = 0.1, range [-1,  1]
    norm >= 100 -> step = 0.05, range [-0.5, 0.5]

Uses a GPU pool with subprocess scheduling and JSON status tracking.

Usage:
    python e8_sweep_eval.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --mode GPT_REWRITE \
        --sizes 1 5 10 25 50 \
        --gpus 0 1 2 3 4 5 6 7

    python e8_sweep_eval.py \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --mode LARGE_MODEL \
        --rewrite_model meta-llama/Llama-3.1-8B-Instruct \
        --sizes 1 5 10 25 50 \
        --gpus 0 1 2 3
"""
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import torch
from datetime import datetime
from pathlib import Path

BASE = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement"

# Layer configurations (from 02_exp_multi_large_pair.py)
MODEL_TO_LAYERS = {
    "Qwen/Qwen2.5-14B-Instruct": [15, 23, 31],
    "Qwen/Qwen2.5-7B-Instruct": [13, 23, 27],
    "Qwen/Qwen2.5-1.5B-Instruct": [2, 3, 4],
    "Qwen/Qwen2.5-0.5B-Instruct": [11, 19, 23],
    "Qwen/Qwen2.5-3B-Instruct": [18, 10, 9, 6],
    "meta-llama/Llama-3.2-1B-Instruct": [13, 14, 8],
}


def compute_lambdas(norm):
    """Compute lambda sweep values based on vector norm.

    norm <  10  -> step = 0.5, range [-5,  5]   (21 values)
    norm 10~100 -> step = 0.1, range [-1,  1]   (21 values)
    norm >= 100 -> step = 0.05, range [-0.5, 0.5] (21 values)
    """
    if norm < 10:
        step, max_lam = 0.5, 5.0
    elif norm < 100:
        step, max_lam = 0.1, 1.0
    else:
        step, max_lam = 0.05, 0.5

    n_steps = int(round(max_lam / step))
    return [round(i * step, 4) for i in range(-n_steps, n_steps + 1)]


def load_norms(vec_dir):
    """Load per-layer norms from vector_norms.json, fallback to steering_vector.pt."""
    norms_path = os.path.join(vec_dir, "vector_norms.json")
    if os.path.exists(norms_path):
        with open(norms_path) as f:
            raw = json.load(f)
        return {int(k): float(v) for k, v in raw.items()}

    sv_path = os.path.join(vec_dir, "steering_vector.pt")
    if not os.path.exists(sv_path):
        return None
    sv = torch.load(sv_path, map_location="cpu", weights_only=False)
    acts = sv.layer_activations if hasattr(sv, "layer_activations") else sv
    norms = {int(k): v.float().norm().item() for k, v in acts.items()}
    with open(norms_path, "w") as f:
        json.dump(norms, f, indent=2)
    return norms


def query_gpu_free_mem():
    """Return {gpu_id: free_mem_mb} via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            encoding="utf-8",
        )
        result = {}
        for i, line in enumerate(out.splitlines()):
            line = line.strip()
            if line:
                result[i] = int(line)
        return result
    except Exception:
        return {}


# ─── Job ─────────────────────────────────────────────────────

class Job:
    def __init__(self, pretrained, calib_size, layer, lam, vec_path,
                 outdir_base, tasks, gen_kwargs, batch_size, limit,
                 lm_eval_model):
        self.pretrained = pretrained
        self.calib_size = calib_size
        self.layer = layer
        self.lam = float(lam)
        self.vec_path = vec_path
        self.tasks = tasks
        self.gen_kwargs = gen_kwargs
        self.batch_size = batch_size
        self.limit = limit
        self.lm_eval_model = lm_eval_model

        self.gpu = None
        self.proc = None
        self.status = "pending"
        self.returncode = None
        self.start_ts = None
        self.end_ts = None

        model_short = pretrained.split("/")[-1]
        lam_tag = f"{self.lam}".replace(".", "p").replace("-", "n")
        self.safe_name = f"{model_short}_N{calib_size}_L{layer}_lam{lam_tag}"
        self.job_id = self.safe_name
        self.outdir = os.path.join(outdir_base, self.safe_name)
        self.stdout_log = os.path.join(self.outdir, "stdout.log")
        self.stderr_log = os.path.join(self.outdir, "stderr.log")
        self._cmd = None

    def build_cmd(self, gpu_id):
        model_args = (
            f"pretrained={self.pretrained},"
            f"dtype=float16,"
            f"steer_layer={self.layer},"
            f"steer_lambda={self.lam},"
            f"steer_vec_path={self.vec_path}"
        )
        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", self.lm_eval_model,
            "--model_args", model_args,
            "--tasks", self.tasks,
            "--device", "cuda:0",
            "--num_fewshot", "0",
            "--batch_size", str(self.batch_size),
            "--gen_kwargs", self.gen_kwargs,
            "--output_path", self.outdir,
            "--log_samples",
            "--apply_chat_template",
        ]
        if self.limit:
            cmd.extend(["--limit", str(self.limit)])
        self._cmd = cmd
        return cmd

    def cmd_str(self):
        return shlex.join(self._cmd) if self._cmd else ""

    def to_record(self):
        dur = None
        if self.start_ts and self.end_ts:
            dur = (self.end_ts - self.start_ts).total_seconds()
        return {
            "job_id": self.job_id,
            "pretrained": self.pretrained,
            "calib_size": self.calib_size,
            "layer": self.layer,
            "lambda": self.lam,
            "gpu": self.gpu,
            "status": self.status,
            "returncode": self.returncode,
            "start_ts": self.start_ts.isoformat() if self.start_ts else None,
            "end_ts": self.end_ts.isoformat() if self.end_ts else None,
            "duration_sec": dur,
            "outdir": self.outdir,
            "cmd": self.cmd_str(),
        }


# ─── Status tracking ────────────────────────────────────────

RUNS_STATE = {}


def load_runs(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "jobs" in data:
            return data["jobs"] if isinstance(data["jobs"], dict) else {}
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return {}


def save_runs(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"updated_at": datetime.now().isoformat(), "jobs": RUNS_STATE},
            f, indent=2, ensure_ascii=False,
        )


# ─── Launch ──────────────────────────────────────────────────

def launch_job(job, gpu_id, runs_path):
    global RUNS_STATE
    job.gpu = gpu_id
    os.makedirs(job.outdir, exist_ok=True)

    cmd = job.build_cmd(gpu_id)
    print(f"[LAUNCH] GPU {gpu_id} -> {job.job_id}")

    job.start_ts = datetime.now()
    job.status = "running"
    RUNS_STATE[job.job_id] = job.to_record()
    save_runs(runs_path)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    stdout_f = open(job.stdout_log, "w")
    stderr_f = open(job.stderr_log, "w")
    proc = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f, env=env)
    job.proc = proc
    job._stdout_f = stdout_f
    job._stderr_f = stderr_f
    return proc


def finish_job(job, ret, runs_path):
    global RUNS_STATE
    job.returncode = ret
    job.end_ts = datetime.now()
    job.status = "done" if ret == 0 else "failed"
    RUNS_STATE[job.job_id] = job.to_record()
    save_runs(runs_path)

    for fh in (getattr(job, "_stdout_f", None), getattr(job, "_stderr_f", None)):
        if fh:
            try:
                fh.close()
            except Exception:
                pass

    tag = "DONE" if ret == 0 else "FAIL"
    print(f"[{tag}] GPU {job.gpu}: N={job.calib_size} L={job.layer} lam={job.lam} rc={ret}")


# ─── Main ────────────────────────────────────────────────────

def main():
    global RUNS_STATE

    parser = argparse.ArgumentParser(description="E8 Calibration Size Sweep")
    parser.add_argument("--model", required=True,
                        help="HF model name, e.g. Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--mode", required=True,
                        choices=["GPT_REWRITE", "LARGE_MODEL"])
    parser.add_argument("--rewrite_model", default=None,
                        help="Rewrite model (LARGE_MODEL mode only)")
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[1, 5, 10, 25, 50])
    parser.add_argument("--gpus", nargs="+", type=int,
                        default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--tasks", default="gsm8k_cot_zeroshot_unified")
    parser.add_argument("--gen_kwargs",
                        default="max_gen_toks=2048,temperature=0,do_sample=False")
    parser.add_argument("--batch_size", default="16")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--lm_eval_model", default="steer_hf")
    parser.add_argument("--min_free_mem_mb", type=int, default=20000)
    parser.add_argument("--max_procs_per_gpu", type=int, default=1)
    parser.add_argument("--vector_base", default=None,
                        help="Override base dir for vectors")
    args = parser.parse_args()

    model_safe = args.model.replace("/", "_")

    # Resolve vector base directory (must match e8_extract_vectors.py output)
    if args.vector_base:
        vec_base = args.vector_base
    else:
        if args.mode == "GPT_REWRITE":
            vec_base = os.path.join(
                BASE, "calibration_ablation", model_safe, "GPT_REWRITE"
            )
        else:
            if not args.rewrite_model:
                parser.error("--rewrite_model required for LARGE_MODEL mode")
            rewrite_safe = args.rewrite_model.replace("/", "_")
            vec_base = os.path.join(
                BASE, "calibration_ablation", model_safe,
                f"LARGE_MODEL_{rewrite_safe}"
            )

    eval_base = os.path.join(vec_base, "eval")
    runs_path = os.path.join(vec_base, "sweep_status.json")

    layers = MODEL_TO_LAYERS.get(args.model)
    if not layers:
        avail = ", ".join(MODEL_TO_LAYERS.keys())
        print(f"[ERROR] No layer config for {args.model}. Available: {avail}")
        sys.exit(1)

    print(f"Model:      {args.model}")
    print(f"Mode:       {args.mode}")
    print(f"Sizes:      {args.sizes}")
    print(f"Layers:     {layers}")
    print(f"GPUs:       {args.gpus}")
    print(f"Vec base:   {vec_base}")
    print(f"Eval dir:   {eval_base}")
    print(f"Status:     {runs_path}")

    # ─── Build job queue ─────────────────────────────────────
    t0 = time.time()
    RUNS_STATE = load_runs(runs_path)
    if RUNS_STATE:
        done_cnt = sum(
            1 for v in RUNS_STATE.values()
            if v.get("status") == "done" and v.get("returncode") == 0
        )
        print(f"[INFO] Loaded history: {len(RUNS_STATE)} records ({done_cnt} done)")

    queue = []
    total_skipped = 0

    for n in args.sizes:
        vec_dir = os.path.join(vec_base, f"vectors_N{n}")
        vec_path = os.path.join(vec_dir, "steering_vector.pt")

        if not os.path.exists(vec_path):
            print(f"[WARN] Vector not found for N={n}: {vec_path}, skipping.")
            continue

        norms = load_norms(vec_dir)
        if norms is None:
            print(f"[WARN] Cannot load norms for N={n}, skipping.")
            continue

        for layer in layers:
            if layer not in norms:
                print(f"[WARN] Layer {layer} not in norms for N={n}, skipping.")
                continue

            norm = norms[layer]
            lambdas = compute_lambdas(norm)
            step_str = "0.5" if norm < 10 else ("0.1" if norm < 100 else "0.05")
            print(
                f"  N={n:3d} L={layer:2d} norm={norm:.4f} "
                f"-> {len(lambdas)} lambdas (step={step_str})"
            )

            for lam in lambdas:
                job = Job(
                    pretrained=args.model,
                    calib_size=n,
                    layer=layer,
                    lam=lam,
                    vec_path=vec_path,
                    outdir_base=eval_base,
                    tasks=args.tasks,
                    gen_kwargs=args.gen_kwargs,
                    batch_size=args.batch_size,
                    limit=args.limit,
                    lm_eval_model=args.lm_eval_model,
                )

                prev = RUNS_STATE.get(job.job_id)
                if (prev and prev.get("status") == "done"
                        and prev.get("returncode") == 0):
                    total_skipped += 1
                    continue

                queue.append(job)

    print(f"\n[INFO] Queued {len(queue)} jobs, skipped {total_skipped} already done.")
    save_runs(runs_path)

    if not queue:
        print("[INFO] Nothing to do.")
        return

    # ─── GPU pool scheduling ─────────────────────────────────
    running = {g: [] for g in args.gpus}

    def on_sigint(sig, frame):
        print("\n[SIGINT] Terminating running jobs...")
        for g, lst in running.items():
            for job_item, proc in lst:
                try:
                    proc.terminate()
                except Exception:
                    pass
                job_item.status = "failed"
                job_item.end_ts = datetime.now()
                RUNS_STATE[job_item.job_id] = job_item.to_record()
        save_runs(runs_path)
        sys.exit(1)

    signal.signal(signal.SIGINT, on_sigint)

    while queue or any(running.values()):
        gpu_free = query_gpu_free_mem()

        # Try to launch new jobs
        while queue:
            best_gpu = None
            best_free = -1

            for g in args.gpus:
                if len(running[g]) >= args.max_procs_per_gpu:
                    continue
                free = gpu_free.get(g, 0)
                if free < args.min_free_mem_mb:
                    continue
                if free > best_free:
                    best_free = free
                    best_gpu = g

            if best_gpu is None:
                break

            job = queue.pop(0)
            proc = launch_job(job, best_gpu, runs_path)
            running[best_gpu].append((job, proc))

        # Poll running jobs
        time.sleep(5)
        for g in list(running.keys()):
            still = []
            for job, proc in running[g]:
                ret = proc.poll()
                if ret is None:
                    still.append((job, proc))
                else:
                    finish_job(job, ret, runs_path)
            running[g] = still

    elapsed = time.time() - t0

    done = sum(
        1 for v in RUNS_STATE.values()
        if v.get("status") == "done" and v.get("returncode") == 0
    )
    failed = sum(1 for v in RUNS_STATE.values() if v.get("status") == "failed")
    print(
        f"\n[ALL DONE] {done} succeeded, {failed} failed. "
        f"Elapsed: {elapsed / 60:.1f} min ({elapsed:.0f} s)"
    )
    print(f"Status file: {runs_path}")


if __name__ == "__main__":
    main()
