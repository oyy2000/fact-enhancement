#!/usr/bin/env python3
"""
E8 formal eval: layer 6 only, lambda = argmax EM on layer 6 from the pilot sweep
(--limit 100), then run lm_eval with --limit 1000 (or custom).

Outputs go to eval_formal_L6_limit{N}/ under the same vec_base as e8_extract_vectors.
Status: formal_L6_limit{N}_status.json (separate from sweep_status.json).

Usage:
    python e8_formal_eval_l6.py --model Qwen/Qwen2.5-3B-Instruct \\
        --mode GPT_REWRITE --gpus 0 1 2 3

    python e8_formal_eval_l6.py --model Qwen/Qwen2.5-3B-Instruct \\
        --mode LARGE_MODEL --rewrite_model Qwen/Qwen2.5-7B-Instruct \\
        --gpus 4 5 6 7

    python e8_formal_eval_l6.py --dry_run  # print chosen lambdas only
"""
import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime

BASE = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement"

JOB_PATTERN = re.compile(
    r"Qwen2\.5-3B-Instruct_N(\d+)_L(\d+)_lam(n?[\dp]+)"
)


def parse_lambda(s):
    s = s.replace("p", ".").replace("n", "-")
    return float(s)


def best_lambda_layer6(eval_dir, layer=6):
    """From pilot sweep dirs, return {N: (lambda, em)} for given layer."""
    by_n = {}
    if not os.path.isdir(eval_dir):
        return by_n
    for entry in os.listdir(eval_dir):
        m = JOB_PATTERN.match(entry)
        if not m:
            continue
        n, lyr = int(m.group(1)), int(m.group(2))
        if lyr != layer:
            continue
        lam = parse_lambda(m.group(3))
        path = os.path.join(eval_dir, entry)
        em = None
        for root, _, files in os.walk(path):
            for fn in files:
                if fn.startswith("results_") and fn.endswith(".json"):
                    try:
                        with open(os.path.join(root, fn)) as f:
                            data = json.load(f)
                        for _, td in data.get("results", {}).items():
                            if "exact_match,flexible-extract" in td:
                                em = td["exact_match,flexible-extract"]
                                break
                    except Exception:
                        pass
                if em is not None:
                    break
            if em is not None:
                break
        if em is None:
            continue
        prev = by_n.get(n)
        if prev is None or em > prev[1]:
            by_n[n] = (lam, em)
    return {n: (lam, em) for n, (lam, em) in by_n.items()}


class FormalJob:
    def __init__(
        self,
        pretrained,
        calib_size,
        layer,
        lam,
        vec_path,
        outdir,
        tasks,
        gen_kwargs,
        batch_size,
        limit,
        lm_eval_model,
        job_id,
    ):
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
        self.job_id = job_id
        self.status = "pending"
        self.returncode = None
        self.gpu = None
        self.proc = None
        self.start_ts = None
        self.end_ts = None
        self.outdir = outdir
        self.stdout_log = os.path.join(outdir, "stdout.log")
        self.stderr_log = os.path.join(outdir, "stderr.log")
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
            "pilot_em": getattr(self, "pilot_em", None),
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


def launch_job(job, gpu_id, runs_path):
    global RUNS_STATE
    job.gpu = gpu_id
    os.makedirs(job.outdir, exist_ok=True)
    job.build_cmd(gpu_id)
    print(f"[LAUNCH] GPU {gpu_id} -> {job.job_id}")
    job.start_ts = datetime.now()
    job.status = "running"
    RUNS_STATE[job.job_id] = job.to_record()
    save_runs(runs_path)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    stdout_f = open(job.stdout_log, "w")
    stderr_f = open(job.stderr_log, "w")
    proc = subprocess.Popen(job._cmd, stdout=stdout_f, stderr=stderr_f, env=env)
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
    print(f"[{tag}] GPU {job.gpu}: {job.job_id} rc={ret}")


def vec_base_for(model_safe, mode, rewrite_model):
    if mode == "GPT_REWRITE":
        return os.path.join(BASE, "calibration_ablation", model_safe, "GPT_REWRITE")
    rewrite_safe = rewrite_model.replace("/", "_")
    return os.path.join(
        BASE, "calibration_ablation", model_safe,
        f"LARGE_MODEL_{rewrite_safe}",
    )


def main():
    global RUNS_STATE

    parser = argparse.ArgumentParser(description="E8 formal L6 eval (limit 1000)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--mode", required=True, choices=["GPT_REWRITE", "LARGE_MODEL"])
    parser.add_argument("--rewrite_model", default=None)
    parser.add_argument("--sizes", nargs="+", type=int, default=None,
                        help="Calibration N (default: all found in pilot eval)")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--tasks", default="gsm8k_cot_zeroshot_unified")
    parser.add_argument(
        "--gen_kwargs",
        default="max_gen_toks=2048,temperature=0,do_sample=False",
    )
    parser.add_argument("--batch_size", default=None,
                        help="Default: 16 GPT_REWRITE, 2 LARGE_MODEL")
    parser.add_argument("--lm_eval_model", default="steer_hf")
    parser.add_argument("--min_free_mem_mb", type=int, default=20000)
    parser.add_argument("--max_procs_per_gpu", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.mode == "LARGE_MODEL" and not args.rewrite_model:
        parser.error("--rewrite_model required for LARGE_MODEL")

    model_safe = args.model.replace("/", "_")
    vec_base = vec_base_for(model_safe, args.mode, args.rewrite_model or "")
    pilot_eval = os.path.join(vec_base, "eval")
    eval_sub = f"eval_formal_L{args.layer}_limit{args.limit}"
    eval_base = os.path.join(vec_base, eval_sub)
    runs_path = os.path.join(vec_base, f"formal_L{args.layer}_limit{args.limit}_status.json")

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = "16" if args.mode == "GPT_REWRITE" else "2"

    best = best_lambda_layer6(pilot_eval, layer=args.layer)
    if not best:
        print(f"[ERROR] No pilot results for layer {args.layer} under {pilot_eval}")
        sys.exit(1)

    sizes = args.sizes if args.sizes is not None else sorted(best.keys())
    print(f"Mode:       {args.mode}")
    print(f"Vec base:   {vec_base}")
    print(f"Pilot eval: {pilot_eval}")
    print(f"Formal out: {eval_base}")
    print(f"Status:     {runs_path}")
    print(f"Layer:      {args.layer}  limit: {args.limit}  batch_size: {batch_size}")
    print("\nChosen lambdas (pilot EM on L6):")
    for n in sizes:
        if n not in best:
            print(f"  N={n}: (no pilot L{args.layer} result, skip)")
            continue
        lam, em = best[n]
        print(f"  N={n:3d}  lambda={lam:>6}  pilot_EM={em:.4f}")

    if args.dry_run:
        return

    RUNS_STATE = load_runs(runs_path)
    queue = []
    model_short = args.model.split("/")[-1]

    for n in sizes:
        if n not in best:
            continue
        lam, pilot_em = best[n]
        vec_path = os.path.join(vec_base, f"vectors_N{n}", "steering_vector.pt")
        if not os.path.exists(vec_path):
            print(f"[WARN] Missing vector: {vec_path}")
            continue
        lam_tag = f"{lam}".replace(".", "p").replace("-", "n")
        job_id = f"{model_short}_formal{args.limit}_N{n}_L{args.layer}_lam{lam_tag}"
        outdir = os.path.join(eval_base, job_id)
        job = FormalJob(
            pretrained=args.model,
            calib_size=n,
            layer=args.layer,
            lam=lam,
            vec_path=vec_path,
            outdir=outdir,
            tasks=args.tasks,
            gen_kwargs=args.gen_kwargs,
            batch_size=batch_size,
            limit=args.limit,
            lm_eval_model=args.lm_eval_model,
            job_id=job_id,
        )
        job.pilot_em = pilot_em
        prev = RUNS_STATE.get(job_id)
        if prev and prev.get("status") == "done" and prev.get("returncode") == 0:
            print(f"[SKIP] already done: {job_id}")
            continue
        queue.append(job)

    print(f"\n[INFO] Queued {len(queue)} formal jobs.")
    if not queue:
        return

    running = {g: [] for g in args.gpus}
    t0 = time.time()

    def on_sigint(sig, frame):
        print("\n[SIGINT] Terminating...")
        for g, lst in running.items():
            for job_item, proc in lst:
                try:
                    proc.terminate()
                except Exception:
                    pass
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
    print(f"\n[ALL DONE] {done} ok, {failed} fail, {elapsed / 60:.1f} min")
    print(f"Status: {runs_path}")


if __name__ == "__main__":
    main()
