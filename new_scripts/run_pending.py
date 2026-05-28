#!/usr/bin/env python3
"""
Idempotent task scheduler for Figure 1 experiments.

Reads tasks.json, finds pending tasks, assigns to free GPUs, runs in parallel,
updates tasks.json with results. Safe to run repeatedly — completed tasks are
skipped, failed tasks are retried.

Usage:
    python new_scripts/run_pending.py                  # run all pending tasks
    python new_scripts/run_pending.py --dry-run        # show plan only
    python new_scripts/run_pending.py --status         # print task status
    python new_scripts/run_pending.py --gpus 0,1,2,3   # use specific GPUs
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from threading import Thread, Lock, Condition

BASE = Path("/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement")
TASKS_JSON = BASE / "new_exps" / "tasks.json"
GSM8K_SCRIPT = BASE / "new_scripts" / "figure1_sampling_vllm.py"
MULTI_SCRIPT = BASE / "new_scripts" / "figure1_multi_dataset_vllm.py"
LOGDIR = BASE / "new_exps" / "figure1_multi_dataset" / "logs"
PYTHON = "/common/home/sl2148/anaconda3/envs/fact_yang/bin/python"

EXPECTED_LINES = {"gsm8k": 1319, "math500": 500, "aime": 30, "amc": 40, "olympiad": 675}

_HF_HOME = str(BASE / "new_exps" / ".cache" / "huggingface")
_HF_TOKEN_PATH = Path(_HF_HOME) / "token"
_HF_TOKEN = _HF_TOKEN_PATH.read_text().strip() if _HF_TOKEN_PATH.is_file() else ""

ENV_EXTRA = {
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_CACHE_ROOT": str(BASE / "new_exps" / ".cache" / "vllm"),
    "XDG_CACHE_HOME": str(BASE / "new_exps" / ".cache"),
    "HF_HOME": _HF_HOME,
    "TRANSFORMERS_CACHE": str(Path(_HF_HOME) / "hub"),
    "HF_TOKEN": _HF_TOKEN,
    "HUGGING_FACE_HUB_TOKEN": _HF_TOKEN,
}

json_lock = Lock()


def get_output_path(model: str, dataset: str) -> Path:
    ms = model.replace("/", "_")
    if dataset == "gsm8k":
        return BASE / "new_exps" / "figure1_sampling_data" / ms / "gsm8k_samples.jsonl"
    return BASE / "new_exps" / "figure1_multi_dataset" / dataset / ms / "samples.jsonl"


def is_output_complete(model: str, dataset: str) -> bool:
    p = get_output_path(model, dataset)
    if not p.is_file():
        return False
    lines = sum(1 for _ in open(p))
    return lines >= EXPECTED_LINES[dataset]


def get_free_gpus(threshold_mib: int = 38000) -> list[int]:
    """Return GPU indices with at least threshold_mib free memory."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        free = []
        for line in out.strip().split("\n"):
            idx, mem = line.split(",")
            if int(mem.strip()) >= threshold_mib:
                free.append(int(idx.strip()))
        return free
    except Exception:
        return []


def load_tasks() -> dict:
    with open(TASKS_JSON) as f:
        return json.load(f)


def save_tasks(data: dict):
    data["updated"] = datetime.now().isoformat()
    with json_lock:
        with open(TASKS_JSON, "w") as f:
            json.dump(data, f, indent=2)


def refresh_done_status(data: dict) -> int:
    """Check output files and mark completed tasks. Returns count of newly done."""
    newly_done = 0
    for t in data["tasks"]:
        if t["status"] != "done" and is_output_complete(t["model"], t["dataset"]):
            t["status"] = "done"
            t["last_run"] = datetime.now().isoformat()
            newly_done += 1
    if newly_done:
        save_tasks(data)
    return newly_done


def print_status(data: dict):
    by_status = defaultdict(list)
    for t in data["tasks"]:
        label = "skip" if t.get("skip") else t["status"]
        by_status[label].append(t)

    total = len(data["tasks"])
    print(f"\n{'='*60}")
    print(f" Task Status  ({total} total)")
    print(f"{'='*60}")
    for status in ["done", "pending", "failed", "running", "skip"]:
        tasks = by_status.get(status, [])
        if not tasks:
            continue
        icon = {"done": "✓", "pending": "○", "failed": "✗", "running": "►", "skip": "⊘"}
        print(f"\n {icon.get(status,'?')} {status.upper()} ({len(tasks)}):")
        for t in tasks:
            short = t["model"].split("/")[-1].replace("-Instruct", "")
            extra = ""
            if t.get("last_error"):
                extra = f"  err: {t['last_error'][:60]}"
            elif t.get("attempts", 0) > 0:
                extra = f"  attempts: {t['attempts']}"
            print(f"   {short:25s} × {t['dataset']:10s} (TP={t['tp']}){extra}")

    print(f"\n{'='*60}")
    done_ct = len(by_status.get("done", []))
    pending_ct = len(by_status.get("pending", []))
    failed_ct = len(by_status.get("failed", []))
    skip_ct = len(by_status.get("skip", []))
    print(f" Summary: {done_ct} done, {pending_ct} pending, {failed_ct} failed, {skip_ct} skipped")
    print(f"{'='*60}\n")


def run_task(task: dict, gpu_ids: list[int], data: dict):
    """Run a single task on specified GPUs. Updates data in-place."""
    model = task["model"]
    dataset = task["dataset"]
    tp = task["tp"]
    gpu_mem = task["gpu_mem"]
    max_len = task.get("max_model_len", 4096)
    short = model.split("/")[-1].replace("-Instruct", "")
    gpu_str = ",".join(str(g) for g in gpu_ids)
    logfile = LOGDIR / f"{dataset}_{short}.log"

    if is_output_complete(model, dataset):
        task["status"] = "done"
        task["last_run"] = datetime.now().isoformat()
        save_tasks(data)
        print(f"  [SKIP] {short} × {dataset} — already complete")
        return

    print(f"  [RUN]  {short} × {dataset}  GPU={gpu_str} TP={tp}  {datetime.now().strftime('%H:%M:%S')}")
    task["status"] = "running"
    task["attempts"] = task.get("attempts", 0) + 1
    task["last_run"] = datetime.now().isoformat()
    save_tasks(data)

    env = {**os.environ, **ENV_EXTRA, "CUDA_VISIBLE_DEVICES": gpu_str}

    swap_space = task.get("swap_space", 16 if task["params_b"] >= 14 else 8)

    if dataset == "gsm8k":
        cmd = [
            PYTHON, str(GSM8K_SCRIPT),
            "--model", model,
            "--tensor_parallel_size", str(tp),
            "--num_samples", "8",
            "--temperature", "0.7",
            "--max_tokens", "2048",
            "--gpu_memory_utilization", str(gpu_mem),
            "--max_model_len", str(max_len),
        ]
    else:
        cmd = [
            PYTHON, str(MULTI_SCRIPT),
            "--model", model,
            "--dataset", dataset,
            "--num_samples", "16",
            "--temperature", "0.7",
            "--max_tokens", "2048",
            "--tensor_parallel_size", str(tp),
            "--max_model_len", str(max_len),
            "--gpu_memory_utilization", str(gpu_mem),
            "--swap_space", str(swap_space),
        ]
        if task.get("enforce_eager", False):
            cmd.append("--enforce_eager")

    try:
        with open(logfile, "w") as lf:
            proc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=7200)
        if proc.returncode == 0 and is_output_complete(model, dataset):
            task["status"] = "done"
            task["last_error"] = None
            print(f"  [DONE] {short} × {dataset}  {datetime.now().strftime('%H:%M:%S')}")
        else:
            task["status"] = "failed"
            try:
                with open(logfile) as lf:
                    lines = lf.readlines()
                    task["last_error"] = lines[-1].strip() if lines else f"exit={proc.returncode}"
            except Exception:
                task["last_error"] = f"exit={proc.returncode}"
            print(f"  [FAIL] {short} × {dataset}  exit={proc.returncode}  see {logfile}")
    except subprocess.TimeoutExpired:
        task["status"] = "failed"
        task["last_error"] = "timeout (2h)"
        print(f"  [FAIL] {short} × {dataset}  TIMEOUT")
    except Exception as e:
        task["status"] = "failed"
        task["last_error"] = str(e)[:200]
        print(f"  [FAIL] {short} × {dataset}  {e}")

    task["last_run"] = datetime.now().isoformat()
    save_tasks(data)


class GpuPool:
    """Thread-safe GPU pool. Workers acquire/release GPUs dynamically."""

    def __init__(self, gpu_ids: list[int], min_free_mib: int = 30000):
        self._available = sorted(gpu_ids)
        self._cond = Condition()
        self._min_free_mib = min_free_mib

    @staticmethod
    def _gpu_free_memory() -> dict[int, int]:
        """Return {gpu_idx: free_mib} for all GPUs."""
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.free",
                 "--format=csv,noheader,nounits"], text=True)
            result = {}
            for line in out.strip().split("\n"):
                idx, mem = line.split(",")
                result[int(idx.strip())] = int(mem.strip())
            return result
        except Exception:
            return {}

    def try_acquire(self, n: int) -> list[int] | None:
        with self._cond:
            if len(self._available) >= n:
                # Check actual free memory on candidate GPUs
                free_mem = self._gpu_free_memory()
                candidates = [g for g in self._available
                              if free_mem.get(g, 0) >= self._min_free_mib]
                if len(candidates) >= n:
                    taken = candidates[:n]
                    for g in taken:
                        self._available.remove(g)
                    return taken
            return None

    def release(self, gpus: list[int]):
        with self._cond:
            self._available.extend(gpus)
            self._available.sort()
            self._cond.notify_all()

    @property
    def free_count(self) -> int:
        with self._cond:
            return len(self._available)


def schedule_and_run(data: dict, available_gpus: list[int], dry_run: bool = False, min_free_mib: int = 30000):
    """Dynamic GPU-pool scheduler: tasks grab GPUs on demand, release when done."""
    pending = [t for t in data["tasks"]
               if t["status"] in ("pending", "failed") and not t.get("skip")]

    if not pending:
        print("\nNo pending tasks to run!")
        return

    # Sort: larger TP first (reduce fragmentation), then smaller params
    pending.sort(key=lambda t: (-t["tp"], t["params_b"], t["dataset"]))

    print(f"\n{'='*60}")
    print(f" Pending: {len(pending)} tasks | GPUs: {available_gpus}")
    print(f"{'='*60}")
    for t in pending:
        short = t["model"].split("/")[-1].replace("-Instruct", "")
        print(f"  {short:25s} × {t['dataset']:10s} TP={t['tp']}")
    print(f"{'='*60}\n")

    if dry_run:
        print("DRY RUN — not executing.")
        return

    pool = GpuPool(available_gpus, min_free_mib=min_free_mib)
    task_list = list(pending)
    task_lock = Lock()
    active_threads: list[Thread] = []

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Scheduler started with {len(available_gpus)} GPUs\n")

    while True:
        # Check if any tasks remain
        with task_lock:
            if not task_list:
                break

        # Scan for a task that fits current free GPUs (prefer larger TP)
        matched_task = None
        matched_gpus = None
        with task_lock:
            for i, t in enumerate(task_list):
                gpus = pool.try_acquire(t["tp"])
                if gpus is not None:
                    matched_task = task_list.pop(i)
                    matched_gpus = gpus
                    break

        if matched_task is None:
            # Clean up finished threads
            active_threads = [t for t in active_threads if t.is_alive()]
            # Log waiting status every 60 seconds
            if not hasattr(schedule_and_run, '_last_wait_log') or time.time() - schedule_and_run._last_wait_log > 60:
                free_mem = GpuPool._gpu_free_memory()
                gpu_status = ", ".join(f"GPU{g}:{free_mem.get(g,0)}MB" for g in available_gpus)
                with task_lock:
                    remaining = len(task_list)
                print(f"  [WAIT] {remaining} tasks pending, need >={min_free_mib}MB free. Current: {gpu_status}")
                schedule_and_run._last_wait_log = time.time()
            time.sleep(15)
            continue

        task = matched_task
        gpus = matched_gpus

        def worker(t=task, g=gpus):
            try:
                run_task(t, g, data)
            finally:
                pool.release(g)

        thread = Thread(target=worker)
        thread.start()
        active_threads.append(thread)

        time.sleep(0.5)  # small gap to avoid race on GPU alloc

    for t in active_threads:
        t.join()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] All tasks finished.")


def main():
    parser = argparse.ArgumentParser(description="Idempotent task scheduler")
    parser.add_argument("--status", action="store_true", help="Print status and exit")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU IDs (default: auto-detect free)")
    parser.add_argument("--retry-failed", action="store_true", help="Reset failed tasks to pending before scheduling")
    parser.add_argument("--min-free", type=int, default=30000, help="Min free GPU memory (MiB) before scheduling (default: 30000)")
    args = parser.parse_args()

    LOGDIR.mkdir(parents=True, exist_ok=True)

    if not TASKS_JSON.is_file():
        print(f"ERROR: {TASKS_JSON} not found. Run task tracker initialization first.")
        sys.exit(1)

    data = load_tasks()

    # Refresh: check output files for any newly completed tasks
    newly = refresh_done_status(data)
    if newly:
        print(f"Found {newly} newly completed tasks from output files.")

    # Always reset stale "running" tasks (leftover from killed runs)
    stale = sum(1 for t in data["tasks"] if t["status"] == "running")
    if stale:
        for t in data["tasks"]:
            if t["status"] == "running":
                t["status"] = "pending"
        save_tasks(data)
        print(f"Reset {stale} stale 'running' tasks to 'pending'.")

    if args.retry_failed:
        count = sum(1 for t in data["tasks"] if t["status"] == "failed")
        for t in data["tasks"]:
            if t["status"] == "failed":
                t["status"] = "pending"
        save_tasks(data)
        print(f"Reset {count} failed tasks to pending.")

    if args.status:
        print_status(data)
        return

    # Determine available GPUs
    if args.gpus:
        available = [int(g) for g in args.gpus.split(",")]
    else:
        available = get_free_gpus()
        if not available:
            print("No free GPUs found (need >=38GB free). Exiting.")
            sys.exit(1)
    print(f"Available GPUs: {available}")

    schedule_and_run(data, available, dry_run=args.dry_run, min_free_mib=args.min_free)

    # Final status
    data = load_tasks()
    refresh_done_status(data)
    print_status(data)


if __name__ == "__main__":
    main()
