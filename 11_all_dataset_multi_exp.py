#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import time
import threading
import argparse
import json
import traceback
from queue import Queue
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# ================= 配置区域 =================

# 1. 定义可用的 GPU ID
AVAILABLE_GPUS = [1]  # 根据你的机器调整

# 2. 定义模型组
MODEL_GROUPS = {
    "qwen": [
        # "Qwen/Qwen2.5-3B-Instruct",
        # "Qwen/Qwen2.5-1.5B-Instruct",
    ],
    "llama": [
        "meta-llama/Llama-3.2-3B-Instruct",
        # "meta-llama/Llama-3.2-1B-Instruct",
    ],
    "all": [],
}
for group_name, models in MODEL_GROUPS.items():
    if group_name != "all":
        MODEL_GROUPS["all"].extend(models)

# 3. 任务列表
TASKS = [
    # "gsm8k_cot_zeroshot_unified_dense",
    "AIME_dense",
    # "AMC_dense",
    # "hendrycks_math_500_dense",
    # "Olympiad",
]

# 4. 基础参数
OUTPUT_PATH = Path("long_cot_vs_short_cot_5")
BACKEND = "hf"
BATCH_SIZE = 32
LIMIT = 1000
MAX_GEN_TOKENS = 2048

# 5. 状态记录文件
STATUS_FILE = OUTPUT_PATH / "experiment_status.json"

# ===========================================

# 线程锁
log_lock = threading.Lock()   # 终端打印锁
json_lock = threading.Lock()  # JSON文件读写锁


class TaskStateManager:
    """
    - 不做 failed 重试（failed 保持 failed）
    - 崩溃恢复：把 running 置回 pending
    """
    def __init__(self, filename: Path, models, tasks):
        self.filename = Path(filename)
        self.data = {}
        self.load_or_initialize(models, tasks)

    def _get_key(self, model, task):
        return f"{model}|{task}"

    def load_or_initialize(self, models, tasks):
        """加载JSON，如果不存在则初始化；如果存在则补充新任务并做崩溃恢复（running->pending）。"""
        with json_lock:
            if self.filename.exists():
                try:
                    with open(self.filename, "r", encoding="utf-8") as f:
                        self.data = json.load(f)
                    print(f"[*] Loaded existing status from {self.filename}")
                except json.JSONDecodeError:
                    print("[!] Warning: JSON file corrupted. Backing up and re-initializing.")
                    bak = str(self.filename) + ".bak"
                    os.rename(self.filename, bak)
                    self.data = {}

            updates = False

            # 初始化或补充新任务
            for model in models:
                for task in tasks:
                    key = self._get_key(model, task)
                    if key not in self.data:
                        self.data[key] = {
                            "model": model,
                            "task": task,
                            "status": "pending",  # pending, running, completed, failed
                            "last_update": "",
                            "error_msg": "",
                        }
                        updates = True

            # 崩溃恢复：running -> pending
            for key, info in self.data.items():
                if info.get("status") == "running":
                    print(f"[-] Resetting crashed task to pending: {info['model']} | {info['task']}")
                    self.data[key]["status"] = "pending"
                    updates = True

            if updates:
                self._save_to_file_locked()

    def _save_to_file_locked(self):
        """写入文件 (必须在 json_lock 内调用)"""
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)

    def get_next_pending_task(self) -> Tuple[Optional[str], Optional[str]]:
        """获取下一个待处理任务，并标记为 running（锁内完成，避免重复领取）"""
        with json_lock:
            for key, info in self.data.items():
                if info.get("status") == "pending":
                    self.data[key]["status"] = "running"
                    self.data[key]["last_update"] = str(datetime.now())
                    self._save_to_file_locked()
                    return info["model"], info["task"]
            return None, None

    def update_status(self, model, task, status, error_msg=""):
        """更新任务状态"""
        key = self._get_key(model, task)
        with json_lock:
            if key in self.data:
                self.data[key]["status"] = status
                self.data[key]["last_update"] = str(datetime.now())
                if error_msg:
                    self.data[key]["error_msg"] = error_msg
                self._save_to_file_locked()


def _stream_to_log(proc: subprocess.Popen, log_f, console_prefix: str = "") -> str:
    """
    实时读取子进程输出（stdout+stderr合并后），逐行写入log并flush。
    返回最后若干行（用于写 error_msg）。
    """
    tail_lines = []
    max_tail = 200  # error_msg 只保留最后 200 行

    # proc.stdout is text stream because text=True
    for line in iter(proc.stdout.readline, ""):
        if line == "":
            break

        # 写log（实时）
        log_f.write(line)
        log_f.flush()

        # 只打印关键状态到终端（默认不逐行打印，避免多线程乱）
        # 如果你确实想逐行打印，取消注释下面这段：
        # with log_lock:
        #     print(f"{console_prefix}{line}", end="")

        tail_lines.append(line)
        if len(tail_lines) > max_tail:
            tail_lines.pop(0)

    return "".join(tail_lines)


def run_task_subprocess(gpu_id: int, model: str, task: str, state_manager: TaskStateManager, args):
    sanitized_model_label = model.replace("/", "__")

    with log_lock:
        print(f"\n[+] [GPU {gpu_id}] STARTED: {model} | {task}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # 给不同任务更合理的 max_gen_toks（按需扩展 key）
    task_max_gen = {
        "gsm8k_cot_zeroshot_unified": 2048,
        "gsm8k_cot_zeroshot": 2048,
        "AMC": 2048,
        "AIME": 2048,
        "hendrycks_math_500": 2048,
        "Olympiad": 2048,
        "gsm8k_cot_zeroshot_unified_dense": MAX_GEN_TOKENS,
        "AIME_dense": MAX_GEN_TOKENS,
        "AMC_dense": MAX_GEN_TOKENS,
        "hendrycks_math_500_dense": MAX_GEN_TOKENS,
        "Olympiad_dense": MAX_GEN_TOKENS,
    }.get(task, MAX_GEN_TOKENS)

    cmd_list = [
        "lm_eval",
        "--model", BACKEND,
        "--tasks", task,
        "--batch_size", str(BATCH_SIZE),
        "--log_samples",
        "--trust_remote_code",
        "--output_path", str(OUTPUT_PATH),
        "--apply_chat_template",
        "--model_args", f"pretrained={model},dtype=float16",
        "--gen_kwargs", f"do_sample=false,temperature=0,max_gen_toks={task_max_gen}",
    ]

    # limit：默认关闭；如果你需要就传 --limit
    if args.limit is not None:
        cmd_list.extend(["--limit", str(args.limit)])

    # 实时 log 文件
    log_dir = OUTPUT_PATH / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{sanitized_model_label}__{task}__gpu{gpu_id}.log"

    try:
        with open(log_file, "w", encoding="utf-8") as lf:
            lf.write(f"[CMD] {' '.join(cmd_list)}\n")
            lf.write(f"[GPU] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n")
            lf.write(f"[START] {datetime.now()}\n\n")
            lf.flush()

            # stderr 合并到 stdout，保证输出按顺序写入同一个 log
            proc = subprocess.Popen(
                cmd_list,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            tail = _stream_to_log(proc, lf, console_prefix=f"[GPU {gpu_id}] ")
            ret = proc.wait()

            lf.write(f"\n[END] {datetime.now()} (returncode={ret})\n")
            lf.flush()

        if ret == 0:
            state_manager.update_status(model, task, "completed")
            with log_lock:
                print(f"[v] [GPU {gpu_id}] FINISHED: {model} | {task} (batch={BATCH_SIZE}, gen={task_max_gen})")
        else:
            error_msg = f"returncode={ret}. See log: {log_file}\n--- tail ---\n{tail}"
            state_manager.update_status(model, task, "failed", error_msg)
            with log_lock:
                print(f"[!] [GPU {gpu_id}] FAILED: {model} | {task}\n    {error_msg}")

    except Exception:
        error_msg = traceback.format_exc()
        state_manager.update_status(model, task, "failed", error_msg)
        with log_lock:
            print(f"[!] [GPU {gpu_id}] CRASHED: {model} | {task}\n{error_msg}")


def worker(gpu_queue: Queue, state_manager: TaskStateManager, args):
    """工作线程"""
    while True:
        # 1. 领取一个任务
        model, task = state_manager.get_next_pending_task()
        if not model:
            break

        # 2. 领取一个 GPU (阻塞等待)
        gpu_id = gpu_queue.get()

        try:
            # 3. 执行任务（不失败重试）
            run_task_subprocess(gpu_id, model, task, state_manager, args)
        finally:
            # 4. 无论如何，归还 GPU
            gpu_queue.put(gpu_id)
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, choices=list(MODEL_GROUPS.keys()), default="all")
    parser.add_argument("--gpus", type=int, nargs="+", default=AVAILABLE_GPUS)
    # 你原来有 LIMIT 常量，这里改成可选参数（默认 None = 不传 --limit）
    parser.add_argument("--limit", type=int, default=None, help="Pass --limit to lm_eval (default: None)")
    args = parser.parse_args()

    selected_models = MODEL_GROUPS[args.group]

    # 1. 初始化状态管理器（不重试 failed）
    state_manager = TaskStateManager(STATUS_FILE, selected_models, TASKS)

    # 打印状态统计
    with json_lock:
        statuses = {}
        for v in state_manager.data.values():
            statuses[v["status"]] = statuses.get(v["status"], 0) + 1
    print("[*] Status summary:", statuses)

    # 2. 初始化 GPU 队列
    gpu_queue = Queue()
    print(f"[*] Using GPUs: {args.gpus}")
    for gpu in args.gpus:
        gpu_queue.put(gpu)

    # 3. 启动线程池
    num_threads = len(args.gpus)
    threads = []

    print(f"[*] Starting {num_threads} worker threads processing '{args.group}' group...")

    for _ in range(num_threads):
        t = threading.Thread(target=worker, args=(gpu_queue, state_manager, args), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("\n[*] All tasks processing finished. Check experiment_status.json for details.")


if __name__ == "__main__":
    main()