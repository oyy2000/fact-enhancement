#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import threading
import argparse
import subprocess
import traceback
from queue import Queue
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

# ================= Steering Config =================
STEERING_CONFIG = {
    "meta-llama/Llama-3.2-3B-Instruct": {
        "LARGE_MODEL": {
            "layers": [22],
            "lambdas": [-0.5],
            "steer_vec_path": (
                "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/"
                "large_model_rewrites_unified_new/"
                "meta-llama_Llama-3.2-3B-Instruct/"
                "vectors_50_paired_meta-llama_Llama-3.1-8B-Instruct/"
                "meta-llama_Llama-3.2-3B-Instruct_applied/"
                "steering_vector.pt"
            ),
        },
    },
}

DEFAULT_TASKS = [
    "Olympiad",
]

DEFAULT_GPUS = [1]

# -----------------------------
# Locks
# -----------------------------
log_lock = threading.Lock()
json_lock = threading.Lock()


def sanitize_model_name(name: str) -> str:
    s = name.replace("/", "__")
    s = re.sub(r"[^A-Za-z0-9_.\-+]+", "_", s)
    return s


def get_free_mem_mb(gpu_id: int) -> Optional[int]:
    """Return free memory (MB) for gpu_id via nvidia-smi; None on error."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_id),
            ],
            stderr=subprocess.DEVNULL,
        )
        free_mb = int(out.decode("utf-8").strip().split("\n")[0])
        return free_mb
    except Exception:
        return None


def wait_for_gpu_memory(gpu_id: int, min_free_gb: float, check_interval: int = 15):
    """Block until GPU free memory (GB) >= min_free_gb."""
    if min_free_gb <= 0:
        return

    target_mb = int(min_free_gb * 1024)
    while True:
        free_mb = get_free_mem_mb(gpu_id)
        if free_mb is None:
            with log_lock:
                print(f"[!] [GPU {gpu_id}] Unable to read free memory; proceeding without check.")
            return

        if free_mb >= target_mb:
            return

        with log_lock:
            print(
                f"[ ] [GPU {gpu_id}] Waiting for free memory {free_mb} MB < target {target_mb} MB; retry in {check_interval}s"
            )
        time.sleep(check_interval)


def _stream_to_log(proc: subprocess.Popen, log_f) -> str:
    """
    Real-time write: read merged stdout/stderr line-by-line, write+flush.
    Returns tail (last N lines) for error_msg.
    """
    tail_lines: List[str] = []
    max_tail = 200

    # proc.stdout is text stream because text=True
    for line in iter(proc.stdout.readline, ""):
        if line == "":
            break
        log_f.write(line)
        log_f.flush()

        tail_lines.append(line)
        if len(tail_lines) > max_tail:
            tail_lines.pop(0)

    return "".join(tail_lines)


class TaskStateManager:
    """
    状态文件 key = model|task|layer|lam
    - 不做 failed 自动重试（failed 保持 failed）
    - 崩溃恢复：running -> pending
    """
    def __init__(self, filename: Path, jobs: List[dict]):
        self.filename = Path(filename)
        self.data: Dict[str, Dict[str, Any]] = {}
        self.load_or_initialize(jobs)

    def _key(self, model: str, task: str, layer: int, lam: float) -> str:
        lam_s = f"{lam:.6f}".rstrip("0").rstrip(".")
        return f"{model}|{task}|L{layer}|lam{lam_s}"

    def load_or_initialize(self, jobs: List[dict]):
        with json_lock:
            if self.filename.exists():
                try:
                    with open(self.filename, "r", encoding="utf-8") as f:
                        self.data = json.load(f)
                    print(f"[*] Loaded status from {self.filename}")
                except json.JSONDecodeError:
                    print("[!] Status JSON corrupted. Backing up and re-initializing.")
                    bak = str(self.filename) + ".bak"
                    os.rename(self.filename, bak)
                    self.data = {}

            updates = False

            # 补充新任务
            for j in jobs:
                k = self._key(j["model"], j["task"], j["layer"], j["lam"])
                if k in self.data and "steer_vec_path" not in self.data[k]:
                    self.data[k]["steer_vec_path"] = j["steer_vec_path"]
                    updates = True

                if k not in self.data:
                    self.data[k] = {
                        "model": j["model"],
                        "task": j["task"],
                        "layer": j["layer"],
                        "lam": j["lam"],
                        "steer_vec_path": j["steer_vec_path"],
                        "status": "pending",
                        "last_update": "",
                        "error_msg": "",
                    }
                    updates = True

            # 崩溃恢复：running -> pending
            for _, info in self.data.items():
                if info.get("status") == "running":
                    info["status"] = "pending"
                    updates = True

            if updates:
                self._save_locked()

    def _save_locked(self):
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def summary(self):
        with json_lock:
            cnt = {}
            for v in self.data.values():
                s = v["status"]
                cnt[s] = cnt.get(s, 0) + 1
            return cnt

    def get_next_pending(self) -> Optional[dict]:
        with json_lock:
            for _, info in self.data.items():
                if info["status"] == "pending":
                    info["status"] = "running"
                    info["last_update"] = str(datetime.now())
                    self._save_locked()
                    return {
                        "model": info["model"],
                        "task": info["task"],
                        "layer": int(info["layer"]),
                        "lam": float(info["lam"]),
                        "steer_vec_path": info["steer_vec_path"],
                    }
        return None

    def update(self, model: str, task: str, layer: int, lam: float, status: str, error_msg: str = ""):
        k = self._key(model, task, layer, lam)
        with json_lock:
            if k in self.data:
                self.data[k]["status"] = status
                self.data[k]["last_update"] = str(datetime.now())
                if error_msg:
                    self.data[k]["error_msg"] = error_msg
                self._save_locked()


def run_one(
    gpu_id: int,
    job: dict,
    experiment_mode: str,
    output_root: Path,
    backend_model_name: str,
    batch_size: int,
    max_gen_default: int,
    limit: Optional[int],
    state: TaskStateManager,
):
    base_model = job["model"]
    task = job["task"]
    layer = int(job["layer"])
    lam = float(job["lam"])
    steer_vec_path = job["steer_vec_path"]

    base_s = sanitize_model_name(base_model)

    task_max_gen = {
        "gsm8k_cot_zeroshot_unified": 2048,
        "gsm8k_cot_zeroshot": 2048,
        "AMC": 2048,
        "AIME": 2048,
        "hendrycks_math_500": 2048,
        "Olympiad": 2048,
        # 你也可以加 dense 的 key；不加就走 default
        "hendrycks_math_500_dense": 2048,
        "Olympiad_dense": 2048,
    }.get(task, max_gen_default)

    lam_tag = f"{lam:.6f}".rstrip("0").rstrip(".")
    outdir = output_root / experiment_mode / base_s / task / f"L{layer}" / f"lam_{lam_tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    with log_lock:
        print(f"\n[+] [GPU {gpu_id}] START: {base_model} | {task} | L{layer} lam={lam} | vec={steer_vec_path}")

    gen_kwargs = f"do_sample=false,temperature=0,max_gen_toks={task_max_gen}"

    model_args = (
        f"pretrained={base_model},"
        f"dtype=float16,"
        f"steer_layer={layer},"
        f"steer_lambda={lam},"
        f"steer_vec_path={steer_vec_path}"
    )

    cmd = [
        "lm_eval",
        "--model", backend_model_name,
        "--model_args", model_args,
        "--tasks", task,
        "--device", "cuda:0",
        "--num_fewshot", "0",
        "--batch_size", str(batch_size),
        "--gen_kwargs", gen_kwargs,
        "--output_path", str(outdir),
        "--log_samples",
        "--trust_remote_code",
        "--apply_chat_template",
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    log_file = outdir / "run.log"

    try:
        with open(log_file, "w", encoding="utf-8") as lf:
            lf.write(f"[CMD] {' '.join(cmd)}\n")
            lf.write(f"[GPU] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n")
            lf.write(f"[START] {datetime.now()}\n\n")
            lf.flush()

            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # merge stderr into stdout for ordered log
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            tail = _stream_to_log(proc, lf)
            ret = proc.wait()

            lf.write(f"\n[END] {datetime.now()} (returncode={ret})\n")
            lf.flush()

        if ret == 0:
            state.update(base_model, task, layer, lam, "completed")
            with log_lock:
                print(f"[v] [GPU {gpu_id}] DONE: {base_model} | {task} | L{layer} lam={lam} (batch={batch_size}, gen={task_max_gen})")
        else:
            msg = f"returncode={ret}, see log: {log_file}\n--- tail ---\n{tail}"
            state.update(base_model, task, layer, lam, "failed", msg)
            with log_lock:
                print(f"[!] [GPU {gpu_id}] FAILED: {base_model} | {task} | L{layer} lam={lam}\n{msg}")

    except Exception:
        msg = traceback.format_exc()
        state.update(base_model, task, layer, lam, "failed", msg)
        with log_lock:
            print(f"[!] [GPU {gpu_id}] CRASHED: {base_model} | {task} | L{layer} lam={lam}\n{msg}")


def worker(gpu_queue: Queue, state: TaskStateManager, args):
    while True:
        job = state.get_next_pending()
        if job is None:
            break

        gpu_id = gpu_queue.get()
        try:
            wait_for_gpu_memory(gpu_id, args.min_free_mem_gb, args.mem_check_interval)
            run_one(
                gpu_id=gpu_id,
                job=job,
                experiment_mode=args.experiment_mode,
                output_root=Path(args.output_root),
                backend_model_name=args.lm_eval_model,
                batch_size=args.batch_size,
                max_gen_default=args.max_gen_toks,
                limit=args.limit,
                state=state,
            )
        finally:
            gpu_queue.put(gpu_id)
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tasks", type=str, nargs="+", default=DEFAULT_TASKS)

    parser.add_argument(
        "--experiment_mode",
        type=str,
        choices=["GPT_REWRITE", "LARGE_MODEL"],
        required=True,
    )
    parser.add_argument("--prompt_style", type=str, default="old",
                        help="Used only in GPT_REWRITE mode, e.g. old")
    parser.add_argument("--rewrite_model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Used to build VECTOR_DIR (rewrite_sanitized). In GPT_REWRITE, set it to whatever folder name you used.")

    parser.add_argument("--gpus", type=int, nargs="+", default=DEFAULT_GPUS)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_gen_toks", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--min_free_mem_gb", type=float, default=20.0,
                        help="Wait until GPU free memory >= this many GB before starting a task (0 to disable)")
    parser.add_argument("--mem_check_interval", type=int, default=15,
                        help="Seconds between GPU memory checks when waiting")

    parser.add_argument("--output_root", type=str, default="steer_runs_6")
    parser.add_argument("--status_file", type=str, default="steer_runs_6/experiment_status.json")

    parser.add_argument("--lm_eval_model", type=str, default="steer_hf",
                        help="lm_eval --model value, e.g. steer_hf")

    args = parser.parse_args()

    print("[*] Using steering config for models:")
    for m in STEERING_CONFIG:
        print("   -", m)

    # 构建 jobs
    jobs: List[dict] = []

    for model, model_cfg in STEERING_CONFIG.items():
        if args.experiment_mode not in model_cfg:
            continue

        cfg = model_cfg[args.experiment_mode]
        vec_path = Path(cfg["steer_vec_path"])
        if not vec_path.exists():
            raise FileNotFoundError(f"Steering vector not found: {vec_path}")

        for task in args.tasks:
            for layer in cfg["layers"]:
                for lam in cfg["lambdas"]:
                    jobs.append({
                        "model": model,
                        "task": task,
                        "layer": int(layer),
                        "lam": float(lam),
                        "steer_vec_path": str(vec_path),
                    })

    status_path = Path(args.status_file)
    state = TaskStateManager(status_path, jobs)

    print("[*] Status summary:", state.summary())
    print(f"[*] Using GPUs: {args.gpus}")
    print(f"[*] Total jobs: {len(jobs)}")

    gpu_queue = Queue()
    for g in args.gpus:
        gpu_queue.put(g)

    threads = []
    for _ in range(len(args.gpus)):
        th = threading.Thread(target=worker, args=(gpu_queue, state, args), daemon=True)
        th.start()
        threads.append(th)

    for th in threads:
        th.join()

    print("\n[*] All done. See status file:", status_path)


if __name__ == "__main__":
    main()