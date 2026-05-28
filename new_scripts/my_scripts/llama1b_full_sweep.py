#!/usr/bin/env python3
"""
Llama-3.2-1B-Instruct: Full layer x lambda sweep on GSM8K (100 samples).
Sweeps ALL 16 layers with lambda range determined by per-layer norm.

Usage:
    python llama1b_full_sweep.py --gpus 0 1 2 3 4 5 6 7
"""
import argparse, json, os, shlex, signal, subprocess, sys, time
from datetime import datetime
from pathlib import Path

BASE = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement"
MODEL = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_SAFE = MODEL.replace("/", "_")
VEC_DIR = os.path.join(BASE, "llama1b_sweep", "vectors", "vectors_N50")
VEC_PATH = os.path.join(VEC_DIR, "steering_vector.pt")
NORMS_PATH = os.path.join(VEC_DIR, "vector_norms.json")
SWEEP_DIR = os.path.join(BASE, "llama1b_sweep")
STATUS_PATH = os.path.join(SWEEP_DIR, "sweep_status.json")
PYTHON = sys.executable
RUNS_STATE = {}


def compute_lambdas(norm):
    if norm < 10:
        step, lo, hi = 0.5, -5.0, 5.0
    elif norm < 100:
        step, lo, hi = 0.1, -1.0, 1.0
    else:
        step, lo, hi = 0.05, -0.5, 0.5
    vals = []
    v = lo
    while v <= hi + 1e-9:
        vals.append(round(v, 4))
        v += step
    return vals, step


def lam_tag(lam):
    return f"lam{f'{lam:.1f}'.replace('-','n').replace('.','p')}"


def save_status():
    with open(STATUS_PATH, "w") as f:
        json.dump({"updated_at": datetime.now().isoformat(), "jobs": RUNS_STATE},
                  f, indent=2, ensure_ascii=False)


def build_jobs(norms, limit, batch_size):
    jobs = []
    for layer_str, norm in sorted(norms.items(), key=lambda x: int(x[0])):
        layer = int(layer_str)
        lambdas, step = compute_lambdas(norm)
        for lam in lambdas:
            tag = f"L{layer}_{lam_tag(lam)}"
            outdir = os.path.join(SWEEP_DIR, tag)
            cmd = (
                f"{PYTHON} -m lm_eval --model steer_hf "
                f"--model_args pretrained={MODEL},dtype=float16,"
                f"steer_layer={layer},steer_lambda={lam},"
                f"steer_vec_path={VEC_PATH} "
                f"--tasks gsm8k_cot_zeroshot_unified "
                f"--device cuda:__GPU__ "
                f"--num_fewshot 0 --batch_size {batch_size} "
                f"--gen_kwargs max_gen_toks=2048,temperature=0,do_sample=False "
                f"--output_path {outdir} "
                f"--log_samples --apply_chat_template --limit {limit}"
            )
            jobs.append({"job_id": tag, "layer": layer, "lambda": lam,
                         "norm": norm, "outdir": outdir, "cmd": cmd})
    return jobs


def run_job(job, gpu, min_free_mem_mb):
    jid = job["job_id"]
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits", f"--id={gpu}"], text=True).strip()
        if int(out.split("\n")[0]) < min_free_mem_mb:
            return None
    except Exception:
        pass
    cmd = job["cmd"].replace("__GPU__", str(gpu))
    outdir = job["outdir"]
    os.makedirs(outdir, exist_ok=True)
    RUNS_STATE[jid] = {**job, "gpu": gpu, "status": "running",
                       "start_ts": datetime.now().isoformat(),
                       "returncode": None, "end_ts": None, "duration_sec": None}
    save_status()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd_local = cmd.replace(f"cuda:{gpu}", "cuda:0")
    with open(os.path.join(outdir, "stdout.log"), "w") as fo, \
         open(os.path.join(outdir, "stderr.log"), "w") as fe:
        proc = subprocess.Popen(shlex.split(cmd_local), stdout=fo, stderr=fe, env=env)
    return proc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--min_free_mem_mb", type=int, default=8000)
    args = parser.parse_args()
    os.makedirs(SWEEP_DIR, exist_ok=True)

    with open(NORMS_PATH) as f:
        norms = json.load(f)
    print(f"Model:  {MODEL}")
    print(f"Layers: {len(norms)} (0-{max(int(k) for k in norms)})")
    print(f"GPUs:   {args.gpus}")
    print(f"Limit:  {args.limit}")
    print(f"Output: {SWEEP_DIR}")

    all_jobs = build_jobs(norms, args.limit, args.batch_size)

    if os.path.exists(STATUS_PATH):
        with open(STATUS_PATH) as f:
            old = json.load(f)
        RUNS_STATE.update(old.get("jobs", {}))
        done_ids = {k for k, v in RUNS_STATE.items()
                    if v.get("status") == "done" and v.get("returncode") == 0}
        print(f"[INFO] Loaded: {len(RUNS_STATE)} records ({len(done_ids)} done)")
    else:
        done_ids = set()

    pending = [j for j in all_jobs if j["job_id"] not in done_ids]
    print(f"[INFO] Total: {len(all_jobs)}, Done: {len(done_ids)}, Pending: {len(pending)}")
    if not pending:
        print("Nothing to do!")
        return

    for ls in sorted(norms.keys(), key=int):
        layer = int(ls)
        norm = norms[ls]
        lambdas, step = compute_lambdas(norm)
        lp = sum(1 for j in pending if j["layer"] == layer)
        print(f"  L{layer}: norm={norm:.4f} -> {len(lambdas)} lambdas (step={step}), {lp} pending")

    gpu_free = {g: None for g in args.gpus}
    t0 = time.time()
    job_idx = 0

    def handle_sigint(sig, frame):
        print("\n[SIGINT] Killing children...")
        for g, info in gpu_free.items():
            if info is not None:
                info[0].kill()
                RUNS_STATE[info[1]]["status"] = "failed"
        save_status()
        sys.exit(1)
    signal.signal(signal.SIGINT, handle_sigint)

    while job_idx < len(pending) or any(v is not None for v in gpu_free.values()):
        for g in list(gpu_free):
            info = gpu_free[g]
            if info is None:
                continue
            proc, jid = info
            rc = proc.poll()
            if rc is not None:
                end = datetime.now()
                start = datetime.fromisoformat(RUNS_STATE[jid]["start_ts"])
                dur = (end - start).total_seconds()
                RUNS_STATE[jid].update({"status": "done" if rc == 0 else "failed",
                                        "returncode": rc, "end_ts": end.isoformat(),
                                        "duration_sec": dur})
                save_status()
                done_n = sum(1 for v in RUNS_STATE.values() if v.get("status") == "done")
                print(f"[DONE] {jid} rc={rc} {dur:.0f}s ({done_n}/{len(all_jobs)}, "
                      f"{(time.time()-t0)/60:.1f}min)")
                gpu_free[g] = None

        for g in list(gpu_free):
            if gpu_free[g] is not None or job_idx >= len(pending):
                continue
            job = pending[job_idx]
            job_idx += 1
            print(f"[LAUNCH] GPU {g} -> {job['job_id']}")
            proc = run_job(job, g, args.min_free_mem_mb)
            if proc:
                gpu_free[g] = (proc, job["job_id"])
            else:
                RUNS_STATE[job["job_id"]]["status"] = "failed"
                save_status()
        time.sleep(2)

    elapsed = time.time() - t0
    done = sum(1 for v in RUNS_STATE.values()
               if v.get("status") == "done" and v.get("returncode") == 0)
    failed = sum(1 for v in RUNS_STATE.values() if v.get("status") == "failed")
    print(f"\n[ALL DONE] {done} succeeded, {failed} failed. "
          f"Elapsed: {elapsed/60:.1f} min ({elapsed:.0f} s)")


if __name__ == "__main__":
    main()
