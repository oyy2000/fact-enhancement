#!/usr/bin/env python3
"""
E11 + E12 with steering vectors trained on non-math reasoning (LogiQA by default; HotpotQA fallback).

Lambdas are placeholders — calibrate on `logiqa` or `longbench_hotpotqa` (see --experiments) before trusting numbers.
"""

import os
import re
import sys
import json
import time
import threading
import argparse
import subprocess
import traceback
from queue import Queue
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

_BASE = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/exps"
_HF_TOKEN_PATH = Path(_BASE).parent / "new_exps" / ".cache" / "huggingface" / "token"
_HF_TOKEN = _HF_TOKEN_PATH.read_text().strip() if _HF_TOKEN_PATH.is_file() else ""

MODEL = "Qwen/Qwen2.5-3B-Instruct"

_GSM8K_FALLBACK = (
    f"{_BASE}/gpt_rewrites_unified_new/Qwen_Qwen2.5-3B-Instruct/vectors_50_old/"
    "Qwen_Qwen2.5-3B-Instruct_applied/steering_vector.pt"
)


def _vec_root_for_suite(suite: str) -> str:
    parent = "logiqa_densesteer" if suite == "logiqa" else "hotpotqa_densesteer"
    return f"{_BASE}/{parent}/vectors/Qwen_Qwen2.5-3B-Instruct"


def make_steer_config(suite: str) -> Dict[str, Any]:
    vr = _vec_root_for_suite(suite)
    dense = f"{vr}/N50_dense_gpt_old/Qwen_Qwen2.5-3B-Instruct_applied/steering_vector.pt"
    infam = f"{vr}/N50_infamily_7b/Qwen_Qwen2.5-3B-Instruct_applied/steering_vector.pt"
    return {
        "BASELINE": {
            "layers": [0],
            "lambdas": [0.0],
            "steer_vec_path": _GSM8K_FALLBACK,
        },
        "LOGIQA_DENSE": {
            "layers": [6],
            "lambdas": [2.0],
            "steer_vec_path": dense,
        },
        "LOGIQA_INFAMILY": {
            "layers": [6],
            "lambdas": [0.45],
            "steer_vec_path": infam,
        },
    }

E11_TASKS = ["mmlu", "gpqa_main_zeroshot"]
E12_TASKS = ["bbh_cot_zeroshot", "longbench_hotpotqa"]

TASK_MAX_GEN = {
    "mmlu": 256,
    "gpqa_main_zeroshot": 256,
    "bbh_cot_zeroshot": 1024,
    "longbench_hotpotqa": 512,
    "longbench_musique": 512,
    "logiqa": 512,
}

DEFAULT_GPUS = [2, 3, 4, 5, 6, 7]

log_lock = threading.Lock()
json_lock = threading.Lock()


def sanitize_model_name(name: str) -> str:
    s = name.replace("/", "__")
    return re.sub(r"[^A-Za-z0-9_.\-+]+", "_", s)


def get_free_mem_mb(gpu_id: int) -> Optional[int]:
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
        return int(out.decode().strip().split("\n")[0])
    except Exception:
        return None


def wait_for_gpu_memory(gpu_id: int, min_free_gb: float, interval: int = 15):
    if min_free_gb <= 0:
        return
    target_mb = int(min_free_gb * 1024)
    while True:
        free_mb = get_free_mem_mb(gpu_id)
        if free_mb is None:
            return
        if free_mb >= target_mb:
            return
        with log_lock:
            print(f"[ ] [GPU {gpu_id}] free={free_mb}MB < {target_mb}MB; waiting {interval}s")
        time.sleep(interval)


def _stream_to_log(proc, log_f) -> str:
    tail: List[str] = []
    for line in iter(proc.stdout.readline, ""):
        if line == "":
            break
        log_f.write(line)
        log_f.flush()
        tail.append(line)
        if len(tail) > 200:
            tail.pop(0)
    return "".join(tail)


class TaskStateManager:
    def __init__(self, filename: Path, jobs: List[dict]):
        self.filename = Path(filename)
        self.data: Dict[str, Dict[str, Any]] = {}
        self.load_or_initialize(jobs)

    def _key(self, job: dict) -> str:
        lam_s = f"{job['lam']:.6f}".rstrip("0").rstrip(".")
        return f"{job['experiment_mode']}|{job['model']}|{job['task']}|L{job['layer']}|lam{lam_s}"

    def load_or_initialize(self, jobs: List[dict]):
        with json_lock:
            if self.filename.exists():
                try:
                    with open(self.filename, "r") as f:
                        self.data = json.load(f)
                    print(f"[*] Loaded status from {self.filename}")
                except json.JSONDecodeError:
                    bak = str(self.filename) + ".bak"
                    os.rename(self.filename, bak)
                    self.data = {}

            updates = False
            for j in jobs:
                k = self._key(j)
                if k not in self.data:
                    self.data[k] = {**j, "status": "pending", "last_update": "", "error_msg": ""}
                    updates = True

            for info in self.data.values():
                if info.get("status") == "running":
                    info["status"] = "pending"
                    updates = True

            if updates:
                self._save_locked()

    def _save_locked(self):
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filename, "w") as f:
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
            for info in self.data.values():
                if info["status"] == "pending":
                    info["status"] = "running"
                    info["last_update"] = str(datetime.now())
                    self._save_locked()
                    return {
                        "experiment_mode": info["experiment_mode"],
                        "model": info["model"],
                        "task": info["task"],
                        "layer": int(info["layer"]),
                        "lam": float(info["lam"]),
                        "steer_vec_path": info["steer_vec_path"],
                    }
        return None

    def update(self, job: dict, status: str, error_msg: str = ""):
        k = self._key(job)
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
    output_root: Path,
    batch_size: int,
    state: TaskStateManager,
    limit: Optional[int],
):
    exp_mode = job["experiment_mode"]
    task = job["task"]
    layer = int(job["layer"])
    lam = float(job["lam"])
    steer_vec_path = job["steer_vec_path"]
    base_s = sanitize_model_name(MODEL)

    task_max_gen = TASK_MAX_GEN.get(task, 1024)
    lam_tag = f"{lam:.6f}".rstrip("0").rstrip(".")
    outdir = output_root / exp_mode / base_s / task / f"L{layer}" / f"lam_{lam_tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if _HF_TOKEN:
        env["HF_TOKEN"] = _HF_TOKEN
        env["HUGGING_FACE_HUB_TOKEN"] = _HF_TOKEN

    with log_lock:
        print(f"\n[+] [GPU {gpu_id}] START: {exp_mode} | {task} | L{layer} lam={lam}")

    gen_kwargs = f"do_sample=false,temperature=0,max_gen_toks={task_max_gen}"
    model_args = (
        f"pretrained={MODEL},"
        f"dtype=float16,"
        f"steer_layer={layer},"
        f"steer_lambda={lam},"
        f"steer_vec_path={steer_vec_path}"
    )

    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "steer_hf",
        "--model_args",
        model_args,
        "--tasks",
        task,
        "--device",
        "cuda:0",
        "--num_fewshot",
        "0",
        "--batch_size",
        str(batch_size),
        "--gen_kwargs",
        gen_kwargs,
        "--output_path",
        str(outdir),
        "--log_samples",
        "--trust_remote_code",
        "--apply_chat_template",
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    log_file = outdir / "run.log"
    try:
        with open(log_file, "w") as lf:
            lf.write(f"[CMD] {' '.join(cmd)}\n[GPU] {gpu_id}\n[START] {datetime.now()}\n\n")
            lf.flush()
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            tail = _stream_to_log(proc, lf)
            ret = proc.wait()
            lf.write(f"\n[END] {datetime.now()} (returncode={ret})\n")

        if ret == 0:
            state.update(job, "completed")
            with log_lock:
                print(f"[v] [GPU {gpu_id}] DONE: {exp_mode} | {task} | L{layer} lam={lam}")
        else:
            msg = f"returncode={ret}, log={log_file}\n--- tail ---\n{tail[-2000:]}"
            state.update(job, "failed", msg)
            with log_lock:
                print(f"[!] [GPU {gpu_id}] FAILED: {exp_mode} | {task} | L{layer} lam={lam}")
    except Exception:
        msg = traceback.format_exc()
        state.update(job, "failed", msg)
        with log_lock:
            print(f"[!] [GPU {gpu_id}] CRASHED: {exp_mode} | {task}")


def worker(gpu_queue: Queue, state: TaskStateManager, args):
    while True:
        job = state.get_next_pending()
        if job is None:
            break
        gpu_id = gpu_queue.get()
        try:
            wait_for_gpu_memory(gpu_id, args.min_free_mem_gb)
            run_one(gpu_id, job, Path(args.output_root), args.batch_size, state, args.limit)
        finally:
            gpu_queue.put(gpu_id)
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(
        description="E11+E12 with LogiQA-trained steering vectors (Qwen2.5-3B-Instruct)"
    )
    parser.add_argument(
        "--steering_suite",
        type=str,
        choices=["logiqa", "hotpotqa"],
        default="logiqa",
        help="Which exps/*_densesteer/vectors tree to load (HotpotQA if LogiQA pipeline unavailable).",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        choices=["E11", "E12", "CALIB_LOGIQA", "CALIB_HOTPOTQA", "ALL"],
        default=["ALL"],
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        choices=["BASELINE", "LOGIQA_DENSE", "LOGIQA_INFAMILY", "ALL"],
        default=["ALL"],
    )
    parser.add_argument("--gpus", type=int, nargs="+", default=DEFAULT_GPUS)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min_free_mem_gb", type=float, default=20.0)
    parser.add_argument(
        "--output_root",
        type=str,
        default="",
        help="Default: exps/steer_runs_e11_e12_logiqa or _hotpotqa from --steering_suite",
    )
    parser.add_argument(
        "--status_file",
        type=str,
        default="",
        help="Default: under output_root/experiment_status.json",
    )
    parser.add_argument(
        "--skip_vec_check",
        action="store_true",
        help="Allow missing LogiQA vector files (jobs will fail at runtime).",
    )
    args = parser.parse_args()

    suite = args.steering_suite
    out_root = args.output_root or (
        "exps/steer_runs_e11_e12_logiqa"
        if suite == "logiqa"
        else "exps/steer_runs_e11_e12_hotpotqa"
    )
    status_file = args.status_file or f"{out_root}/experiment_status.json"
    args.output_root = out_root
    args.status_file = status_file

    steering_config = make_steer_config(suite)

    experiments = list(args.experiments)
    if "ALL" in experiments:
        experiments = ["E11", "E12"]

    modes = list(args.modes)
    if "ALL" in modes:
        modes = ["BASELINE", "LOGIQA_DENSE", "LOGIQA_INFAMILY"]

    tasks = []
    if "CALIB_LOGIQA" in experiments:
        if len(experiments) != 1:
            raise SystemExit("Use --experiments CALIB_LOGIQA alone (no E11/E12 in same run).")
        tasks = ["logiqa"]
    elif "CALIB_HOTPOTQA" in experiments:
        if len(experiments) != 1:
            raise SystemExit("Use --experiments CALIB_HOTPOTQA alone (no E11/E12 in same run).")
        tasks = ["longbench_hotpotqa"]
    else:
        if "E11" in experiments:
            tasks.extend(E11_TASKS)
        if "E12" in experiments:
            tasks.extend(E12_TASKS)

    print(f"[*] Model: {MODEL}")
    print(f"[*] Steering suite: {suite} (vectors under {_vec_root_for_suite(suite)})")
    print(f"[*] Experiments: {experiments}")
    print(f"[*] Modes: {modes}")
    print(f"[*] Tasks: {tasks}")
    print(f"[*] GPUs: {args.gpus}")

    jobs: List[dict] = []
    for mode in modes:
        cfg = steering_config[mode]
        vec_path = Path(cfg["steer_vec_path"])
        if not vec_path.is_file() and not args.skip_vec_check:
            raise FileNotFoundError(
                f"Steering vector not found: {vec_path}\n"
                "Run run_logiqa_densesteer_pipeline.sh first, or pass --skip_vec_check."
            )
        for task in tasks:
            for layer in cfg["layers"]:
                for lam in cfg["lambdas"]:
                    jobs.append(
                        {
                            "experiment_mode": mode,
                            "model": MODEL,
                            "task": task,
                            "layer": int(layer),
                            "lam": float(lam),
                            "steer_vec_path": str(vec_path),
                        }
                    )

    status_path = Path(status_file)
    state = TaskStateManager(status_path, jobs)

    print(f"[*] Total jobs: {len(jobs)}")
    print(f"[*] Status: {state.summary()}")

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

    print(f"\n[*] All done. Status: {state.summary()}")
    print(f"[*] See: {status_path}")


if __name__ == "__main__":
    main()
