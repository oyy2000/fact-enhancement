#!/usr/bin/env python3
"""
E0-E3 Sweep Evaluation: Layer-6 lambda sweep on GSM8K for all ablation variants.

For each experiment (E0, E1, E2, E3, E3.2), loads the pre-extracted steering
vector, reads the layer-6 norm, computes a lambda sweep range, and runs
lm_eval with steer_hf for each lambda value.

Lambda ranges (same as e8_sweep_eval.py):
    norm <  10  -> step = 0.5, range [-5,  5]
    norm 10~100 -> step = 0.1, range [-1,  1]
    norm >= 100 -> step = 0.05, range [-0.5, 0.5]

Usage:
    python e0_e3_sweep_eval.py --gpus 0 1 2 3 4 5 6 7
    python e0_e3_sweep_eval.py --experiments e0 e2 --gpus 0 1
    python e0_e3_sweep_eval.py --experiments e3 --gpus 0 --limit 500
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
CONTROL_DIR = os.path.join(BASE, "control_experiments", "Qwen_Qwen2.5-3B-Instruct")
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
TARGET_LAYER = 6

EXPERIMENT_NAMES = ["e0", "e1", "e2", "e3", "e3.2"]


def compute_lambdas(norm):
    if norm < 10:
        step, max_lam = 0.5, 5.0
    elif norm < 100:
        step, max_lam = 0.1, 1.0
    else:
        step, max_lam = 0.05, 0.5
    n_steps = int(round(max_lam / step))
    return [round(i * step, 4) for i in range(-n_steps, n_steps + 1)]


def load_norms(vec_dir):
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


class Job:
    def __init__(self, experiment, layer, lam, vec_path, outdir_base,
                 tasks, gen_kwargs, batch_size, limit, lm_eval_model):
        self.experiment = experiment
        self.pretrained = MODEL_NAME
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

        lam_tag = f"{self.lam}".replace(".", "p").replace("-", "n")
        self.safe_name = f"{experiment}_L{layer}_lam{lam_tag}"
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
            "experiment": self.experiment,
            "pretrained": self.pretrained,
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
    dur = ""
    if job.start_ts and job.end_ts:
        dur = f" ({(job.end_ts - job.start_ts).total_seconds():.0f}s)"
    print(f"[{tag}] GPU {job.gpu}: {job.experiment} L={job.layer} lam={job.lam} rc={ret}{dur}")


def main():
    global RUNS_STATE

    parser = argparse.ArgumentParser(description="E0-E3 Lambda Sweep on GSM8K")
    parser.add_argument("--experiments", nargs="+", default=EXPERIMENT_NAMES,
                        choices=EXPERIMENT_NAMES,
                        help="Which experiments to sweep (default: all)")
    parser.add_argument("--layer", type=int, default=TARGET_LAYER)
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
    args = parser.parse_args()

    eval_base = os.path.join(CONTROL_DIR, "sweep_eval")
    runs_path = os.path.join(CONTROL_DIR, "sweep_status.json")

    print(f"Model:       {MODEL_NAME}")
    print(f"Layer:       {args.layer}")
    print(f"Experiments: {args.experiments}")
    print(f"GPUs:        {args.gpus}")
    print(f"Eval dir:    {eval_base}")
    print(f"Status:      {runs_path}")

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

    for exp_name in args.experiments:
        vec_dir = os.path.join(CONTROL_DIR, f"vectors_{exp_name}")
        vec_path = os.path.join(vec_dir, "steering_vector.pt")

        if not os.path.exists(vec_path):
            print(f"[WARN] Vector not found for {exp_name}: {vec_path}, skipping.")
            continue

        norms = load_norms(vec_dir)
        if norms is None or args.layer not in norms:
            print(f"[WARN] Layer {args.layer} norm not available for {exp_name}, skipping.")
            continue

        norm = norms[args.layer]
        lambdas = compute_lambdas(norm)
        step_str = "0.5" if norm < 10 else ("0.1" if norm < 100 else "0.05")
        print(f"  {exp_name}: L{args.layer} norm={norm:.4f} -> {len(lambdas)} lambdas (step={step_str})")

        for lam in lambdas:
            job = Job(
                experiment=exp_name,
                layer=args.layer,
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
            if prev and prev.get("status") == "done" and prev.get("returncode") == 0:
                total_skipped += 1
                continue

            queue.append(job)

    print(f"\n[INFO] Queued {len(queue)} jobs, skipped {total_skipped} already done.")
    save_runs(runs_path)

    if not queue:
        print("[INFO] Nothing to do.")
        return

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
