#!/usr/bin/env python3
"""
LogiQA Sweep + Eval: two-phase pipeline for steering vector calibration.

Phase 1 (sweep):  All layers × lambdas with --limit 50 on logiqa_gen (generate CoT + extract Final Answer acc).
Phase 2 (eval):   Best (layer, lambda) per vector with --limit 400 on logiqa_gen.

Lambda sweep: norm-adaptive (same as e8_sweep_eval.py).
    norm <  10  -> step = 0.5, range [-5,  5]
    norm 10~100 -> step = 0.1, range [-1,  1]
    norm >= 100 -> step = 0.05, range [-0.5, 0.5]

Usage:
    # Sweep (default limit=50)
    python logiqa_sweep_eval.py sweep \
        --vectors v1_dense v1_infamily v2_dense v2_infamily \
        --gpus 0 1 2 3 4 5 6 7

    # Eval best configs (default limit=400)
    python logiqa_sweep_eval.py eval \
        --vectors v1_dense v1_infamily v2_dense v2_infamily \
        --gpus 0 1 2 3

    # Custom layers
    python logiqa_sweep_eval.py sweep --layers 6 10 18 --gpus 0 1
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
EXPS = os.path.join(BASE, "exps")

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
ALL_LAYERS = list(range(19))  # Qwen2.5-3B has 19 transformer layers (0..18)
TASK = "logiqa_gen"

# ── Vector registry ──────────────────────────────────────────
VECTOR_REGISTRY = {
    "v1_dense": os.path.join(
        EXPS, "logiqa_densesteer/vectors/Qwen_Qwen2.5-3B-Instruct/"
        "N50_dense_gpt_old/Qwen_Qwen2.5-3B-Instruct_applied"
    ),
    "v1_infamily": os.path.join(
        EXPS, "logiqa_densesteer/vectors/Qwen_Qwen2.5-3B-Instruct/"
        "N50_infamily_7b/Qwen_Qwen2.5-3B-Instruct_applied"
    ),
    "v2_dense": os.path.join(
        EXPS, "logiqa_densesteer_v2/vectors/N50_dense/"
        "Qwen_Qwen2.5-3B-Instruct_applied"
    ),
    "v2_infamily": os.path.join(
        EXPS, "logiqa_densesteer_v2/vectors/N50_infamily/"
        "Qwen_Qwen2.5-3B-Instruct_applied"
    ),
}


# ── Lambda computation ───────────────────────────────────────

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


# ── GPU helpers ──────────────────────────────────────────────

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


# ── Job ──────────────────────────────────────────────────────

class Job:
    def __init__(self, vec_name, layer, lam, vec_path, outdir_base,
                 batch_size, limit, lm_eval_model):
        self.vec_name = vec_name
        self.pretrained = MODEL_NAME
        self.layer = layer
        self.lam = float(lam)
        self.vec_path = vec_path
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
        self.safe_name = f"{vec_name}_L{layer}_lam{lam_tag}"
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
        gen_kwargs = "do_sample=false,temperature=0,max_gen_toks=1024"
        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", self.lm_eval_model,
            "--model_args", model_args,
            "--tasks", TASK,
            "--device", "cuda:0",
            "--num_fewshot", "0",
            "--batch_size", str(self.batch_size),
            "--gen_kwargs", gen_kwargs,
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
            "vec_name": self.vec_name,
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


# ── Status tracking ──────────────────────────────────────────

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


# ── Launch / finish ──────────────────────────────────────────

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
    print(f"[{tag}] GPU {job.gpu}: {job.vec_name} L={job.layer} lam={job.lam} rc={ret}{dur}")


# ── Scheduler ────────────────────────────────────────────────

def run_queue(queue, gpus, min_free_mem_mb, max_procs_per_gpu, runs_path):
    global RUNS_STATE
    running = {g: [] for g in gpus}

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
            for g in gpus:
                if len(running[g]) >= max_procs_per_gpu:
                    continue
                free = gpu_free.get(g, 0)
                if free < min_free_mem_mb:
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


# ── Sweep result parsing ─────────────────────────────────────

def parse_sweep_results(eval_base, vectors):
    """Parse lm_eval result JSONs to find best (layer, lambda) per vector."""
    best = {}  # vec_name -> {"layer": L, "lam": lam, "acc": acc}

    for vec_name in vectors:
        best_acc = -1.0
        best_cfg = None

        for entry in os.listdir(eval_base):
            if not entry.startswith(vec_name + "_L"):
                continue
            result_dir = os.path.join(eval_base, entry)
            # lm_eval writes results under result_dir/<model_name>/results.json
            for root, dirs, files in os.walk(result_dir):
                for fname in files:
                    if fname == "results.json":
                        fpath = os.path.join(root, fname)
                        try:
                            with open(fpath) as f:
                                data = json.load(f)
                            # lm_eval results format
                            results = data.get("results", {})
                            task_result = results.get(TASK, {})
                            acc = task_result.get("exact_match,get-answer",
                                  task_result.get("exact_match,none",
                                  task_result.get("exact_match",
                                  task_result.get("acc,none",
                                  task_result.get("acc", -1)))))
                            if acc > best_acc:
                                best_acc = acc
                                # Parse layer and lambda from entry name
                                parts = entry.split("_L")
                                layer_lam = parts[-1]  # e.g. "6_lam2p0"
                                ll_parts = layer_lam.split("_lam")
                                layer = int(ll_parts[0])
                                lam_str = ll_parts[1].replace("p", ".").replace("n", "-")
                                lam = float(lam_str)
                                best_cfg = {"layer": layer, "lam": lam, "acc": best_acc}
                        except Exception:
                            continue

        if best_cfg:
            best[vec_name] = best_cfg
            print(f"  {vec_name}: best L={best_cfg['layer']} lam={best_cfg['lam']:.2f} acc={best_cfg['acc']:.4f}")
        else:
            print(f"  {vec_name}: no results found")

    return best


# ── Main ─────────────────────────────────────────────────────

def main():
    global RUNS_STATE

    parser = argparse.ArgumentParser(
        description="LogiQA Sweep + Eval for steering vector calibration"
    )
    sub = parser.add_subparsers(dest="phase", required=True)

    # Common args
    for name, help_text in [("sweep", "Phase 1: lambda sweep"), ("eval", "Phase 2: eval best configs")]:
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--vectors", nargs="+",
                        default=list(VECTOR_REGISTRY.keys()),
                        choices=list(VECTOR_REGISTRY.keys()),
                        help="Which vectors to use")
        sp.add_argument("--gpus", nargs="+", type=int,
                        default=[0, 1, 2, 3, 4, 5, 6, 7])
        sp.add_argument("--batch_size", type=int, default=16)
        sp.add_argument("--lm_eval_model", default="steer_hf")
        sp.add_argument("--min_free_mem_mb", type=int, default=20000)
        sp.add_argument("--max_procs_per_gpu", type=int, default=1)
        sp.add_argument("--output_base", default=os.path.join(EXPS, "logiqa_sweep"))

    # Sweep-specific
    sweep_p = sub.choices["sweep"]
    sweep_p.add_argument("--layers", nargs="+", type=int, default=ALL_LAYERS,
                         help="Layers to sweep (default: all 0..18)")
    sweep_p.add_argument("--limit", type=int, default=50,
                         help="Sample limit for sweep (default: 50)")

    # Eval-specific
    eval_p = sub.choices["eval"]
    eval_p.add_argument("--limit", type=int, default=400,
                        help="Sample limit for eval (default: 400)")
    eval_p.add_argument("--sweep_dir", default=None,
                        help="Override sweep results dir (default: output_base/sweep)")

    args = parser.parse_args()

    if args.phase == "sweep":
        do_sweep(args)
    else:
        do_eval(args)


def do_sweep(args):
    global RUNS_STATE

    eval_base = os.path.join(args.output_base, "sweep")
    runs_path = os.path.join(args.output_base, "sweep_status.json")

    print(f"[*] Phase: SWEEP (limit={args.limit})")
    print(f"[*] Task: {TASK} (multiple_choice acc)")
    print(f"[*] Model: {MODEL_NAME}")
    print(f"[*] Vectors: {args.vectors}")
    print(f"[*] Layers: {args.layers}")
    print(f"[*] GPUs: {args.gpus}")
    print(f"[*] Output: {eval_base}")

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

    for vec_name in args.vectors:
        vec_dir = VECTOR_REGISTRY[vec_name]
        vec_path = os.path.join(vec_dir, "steering_vector.pt")

        if not os.path.exists(vec_path):
            print(f"[WARN] Vector not found: {vec_path}, skipping {vec_name}.")
            continue

        norms = load_norms(vec_dir)
        if norms is None:
            print(f"[WARN] Cannot load norms for {vec_name}, skipping.")
            continue

        for layer in args.layers:
            if layer not in norms:
                continue

            norm = norms[layer]
            lambdas = compute_lambdas(norm)
            step_str = "0.5" if norm < 10 else ("0.1" if norm < 100 else "0.05")
            print(f"  {vec_name} L={layer:2d} norm={norm:.4f} -> {len(lambdas)} lambdas (step={step_str})")

            for lam in lambdas:
                job = Job(
                    vec_name=vec_name,
                    layer=layer,
                    lam=lam,
                    vec_path=vec_path,
                    outdir_base=eval_base,
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
        # Still print summary
        print("\n[*] Sweep summary:")
        parse_sweep_results(eval_base, args.vectors)
        return

    run_queue(queue, args.gpus, args.min_free_mem_mb, args.max_procs_per_gpu, runs_path)

    elapsed = time.time() - t0
    done = sum(
        1 for v in RUNS_STATE.values()
        if v.get("status") == "done" and v.get("returncode") == 0
    )
    failed = sum(1 for v in RUNS_STATE.values() if v.get("status") == "failed")
    print(f"\n[ALL DONE] {done} succeeded, {failed} failed. Elapsed: {elapsed / 60:.1f} min")
    print(f"Status: {runs_path}")

    # Print best configs
    print("\n[*] Best configs per vector:")
    best = parse_sweep_results(eval_base, args.vectors)
    summary_path = os.path.join(args.output_base, "sweep_best.json")
    with open(summary_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Saved to {summary_path}")


def do_eval(args):
    global RUNS_STATE

    sweep_dir = args.sweep_dir or os.path.join(args.output_base, "sweep")
    eval_base = os.path.join(args.output_base, "eval")
    runs_path = os.path.join(args.output_base, "eval_status.json")

    print(f"[*] Phase: EVAL (limit={args.limit})")
    print(f"[*] Task: {TASK} (multiple_choice acc)")
    print(f"[*] Model: {MODEL_NAME}")
    print(f"[*] Vectors: {args.vectors}")
    print(f"[*] GPUs: {args.gpus}")

    # Parse sweep results to find best configs
    print("\n[*] Reading sweep results from:", sweep_dir)
    best = parse_sweep_results(sweep_dir, args.vectors)

    if not best:
        print("[ERROR] No sweep results found. Run 'sweep' phase first.")
        sys.exit(1)

    t0 = time.time()
    RUNS_STATE = load_runs(runs_path)

    queue = []
    total_skipped = 0

    # Also add baseline (lambda=0) for comparison
    for vec_name in args.vectors:
        if vec_name not in best:
            print(f"[WARN] No sweep result for {vec_name}, skipping eval.")
            continue

        cfg = best[vec_name]
        vec_dir = VECTOR_REGISTRY[vec_name]
        vec_path = os.path.join(vec_dir, "steering_vector.pt")

        # Best config
        job = Job(
            vec_name=vec_name,
            layer=cfg["layer"],
            lam=cfg["lam"],
            vec_path=vec_path,
            outdir_base=eval_base,
            batch_size=args.batch_size,
            limit=args.limit,
            lm_eval_model=args.lm_eval_model,
        )
        prev = RUNS_STATE.get(job.job_id)
        if prev and prev.get("status") == "done" and prev.get("returncode") == 0:
            total_skipped += 1
        else:
            queue.append(job)

        # Baseline (lambda=0)
        baseline_job = Job(
            vec_name=f"{vec_name}_baseline",
            layer=cfg["layer"],
            lam=0.0,
            vec_path=vec_path,
            outdir_base=eval_base,
            batch_size=args.batch_size,
            limit=args.limit,
            lm_eval_model=args.lm_eval_model,
        )
        prev = RUNS_STATE.get(baseline_job.job_id)
        if prev and prev.get("status") == "done" and prev.get("returncode") == 0:
            total_skipped += 1
        else:
            queue.append(baseline_job)

    print(f"\n[INFO] Queued {len(queue)} eval jobs, skipped {total_skipped} already done.")
    save_runs(runs_path)

    if not queue:
        print("[INFO] Nothing to do.")
        return

    run_queue(queue, args.gpus, args.min_free_mem_mb, args.max_procs_per_gpu, runs_path)

    elapsed = time.time() - t0
    done = sum(
        1 for v in RUNS_STATE.values()
        if v.get("status") == "done" and v.get("returncode") == 0
    )
    failed = sum(1 for v in RUNS_STATE.values() if v.get("status") == "failed")
    print(f"\n[ALL DONE] {done} succeeded, {failed} failed. Elapsed: {elapsed / 60:.1f} min")

    # Print final eval results
    print("\n[*] Eval results:")
    parse_sweep_results(eval_base, args.vectors)
    parse_sweep_results(eval_base, [f"{v}_baseline" for v in args.vectors])


if __name__ == "__main__":
    main()
