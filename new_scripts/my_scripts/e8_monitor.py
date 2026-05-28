#!/usr/bin/env python3
"""
E8 Monitor: Continuously monitors and auto-restarts E8 sweep jobs.
Run with: nohup /path/to/python -u e8_monitor.py > e8_monitor.log 2>&1 &
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

BASE = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement"
PYTHON = sys.executable
SWEEP_SCRIPT = os.path.join(BASE, "new_scripts/my_scripts/e8_sweep_eval.py")

SWEEPS = [
    {
        "name": "GPT_REWRITE",
        "status_file": os.path.join(
            BASE, "calibration_ablation/Qwen_Qwen2.5-3B-Instruct/GPT_REWRITE/sweep_status.json"
        ),
        "args": [
            "--model", "Qwen/Qwen2.5-3B-Instruct",
            "--mode", "GPT_REWRITE",
            "--sizes", "1", "5", "10", "25", "50",
            "--limit", "100", "--batch_size", "16",
        ],
        "log": os.path.join(BASE, "new_scripts/my_scripts/logs/e8_sweep_gpt_rewrite.log"),
        "total": 420,
        "proc": None,
        "preferred_gpu_count": 2,
    },
    {
        "name": "LARGE_MODEL",
        "status_file": os.path.join(
            BASE, "calibration_ablation/Qwen_Qwen2.5-3B-Instruct/"
            "LARGE_MODEL_Qwen_Qwen2.5-7B-Instruct/sweep_status.json"
        ),
        "args": [
            "--model", "Qwen/Qwen2.5-3B-Instruct",
            "--mode", "LARGE_MODEL",
            "--rewrite_model", "Qwen/Qwen2.5-7B-Instruct",
            "--sizes", "1", "5", "10", "25", "50",
            "--limit", "100", "--batch_size", "16",
        ],
        "log": os.path.join(BASE, "new_scripts/my_scripts/logs/e8_sweep_large_model.log"),
        "total": 420,
        "proc": None,
        "preferred_gpu_count": 6,
    },
]

CHECK_INTERVAL = 120  # seconds between checks
STALE_THRESHOLD = 300  # consider dead if status not updated for 5 min


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def get_free_gpus(min_free_mb=30000):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        )
        free = []
        for line in out.strip().split("\n"):
            parts = line.split(",")
            if len(parts) == 2:
                idx, mem = int(parts[0].strip()), int(parts[1].strip())
                if mem > min_free_mb:
                    free.append(idx)
        return free
    except Exception as e:
        log(f"  nvidia-smi error: {e}")
        return []


def read_status(path):
    """Return (done, running, failed, last_update_dt)."""
    if not os.path.exists(path):
        return 0, 0, 0, None
    try:
        with open(path) as f:
            data = json.load(f)
        jobs = data.get("jobs", {})
        done = sum(1 for v in jobs.values()
                   if v.get("status") == "done" and v.get("returncode") == 0)
        running = sum(1 for v in jobs.values()
                      if v.get("status") == "running")
        failed = sum(1 for v in jobs.values()
                     if v.get("status") == "failed")
        ts = data.get("updated_at", "")
        try:
            dt = datetime.fromisoformat(ts)
        except Exception:
            dt = None
        return done, running, failed, dt
    except Exception as e:
        log(f"  Error reading {path}: {e}")
        return 0, 0, 0, None


def is_process_alive(proc):
    if proc is None:
        return False
    return proc.poll() is None


def kill_sweep_children(name):
    """Kill lm_eval processes belonging to this sweep mode."""
    try:
        subprocess.run(
            ["pkill", "-f", f"lm_eval.*{name}"],
            timeout=5, capture_output=True,
        )
    except Exception:
        pass


def launch_sweep(sweep, gpus):
    kill_sweep_children(sweep["name"])
    time.sleep(3)

    gpu_args = ["--gpus"] + [str(g) for g in gpus]
    cmd = [PYTHON, "-u", SWEEP_SCRIPT] + sweep["args"] + gpu_args

    log_f = open(sweep["log"], "w")
    env = os.environ.copy()

    log(f"  Launching {sweep['name']} on GPUs {gpus}")
    log(f"  CMD: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT,
        env=env, cwd=BASE,
    )
    sweep["proc"] = proc
    return proc


def main():
    log("E8 Monitor started.")
    log(f"Check interval: {CHECK_INTERVAL}s, stale threshold: {STALE_THRESHOLD}s")

    while True:
        all_done = True

        free_gpus = get_free_gpus()
        log(f"Free GPUs (>30GB): {free_gpus}")

        for sweep in SWEEPS:
            done, running, failed, last_dt = read_status(sweep["status_file"])
            total = sweep["total"]
            remaining = total - done

            log(f"{sweep['name']}: {done}/{total} done, {running} running, "
                f"{failed} failed, {remaining} remaining")

            if remaining <= 0:
                log(f"  {sweep['name']} COMPLETE!")
                if is_process_alive(sweep["proc"]):
                    sweep["proc"].wait()
                    sweep["proc"] = None
                continue

            all_done = False

            # Check if sweep is alive
            alive = is_process_alive(sweep["proc"])
            stale = False
            if last_dt:
                age = (datetime.now() - last_dt).total_seconds()
                if age > STALE_THRESHOLD:
                    stale = True
                    log(f"  Status file stale ({age:.0f}s old)")

            if not alive or stale:
                log(f"  {sweep['name']} is {'dead' if not alive else 'stale'}, restarting...")

                if alive:
                    sweep["proc"].terminate()
                    try:
                        sweep["proc"].wait(timeout=10)
                    except Exception:
                        sweep["proc"].kill()
                    sweep["proc"] = None

                # Allocate GPUs
                n_gpus = min(sweep["preferred_gpu_count"], len(free_gpus))
                if n_gpus == 0:
                    log(f"  No free GPUs! Will retry next cycle.")
                    continue

                gpus = free_gpus[:n_gpus]
                free_gpus = free_gpus[n_gpus:]

                launch_sweep(sweep, gpus)
            else:
                log(f"  {sweep['name']} alive (PID={sweep['proc'].pid})")

        if all_done:
            log("ALL SWEEPS COMPLETE!")
            break

        log(f"Sleeping {CHECK_INTERVAL}s...")
        time.sleep(CHECK_INTERVAL)

    # Final summary
    log("\n" + "=" * 60)
    log("FINAL SUMMARY")
    log("=" * 60)
    for sweep in SWEEPS:
        done, running, failed, _ = read_status(sweep["status_file"])
        log(f"{sweep['name']}: {done}/{sweep['total']} done, {failed} failed")
    log("Monitor exiting.")


if __name__ == "__main__":
    main()
