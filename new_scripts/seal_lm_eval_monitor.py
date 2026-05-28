#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-score SEAL predictions.jsonl with lm-eval-harness tasks (same metrics as 02_exp_multi_large_pair).

- gsm8k_cot_zeroshot_unified for GSM8K test runs under SEAL/results/.../GSM/...
- math_500_cot_zeroshot for MATH-500 under SEAL/results/.../MATH500/...

Uses custom model `seal_jsonl` (lm_eval.models.seal_jsonl_lm) to replay generations.

GSM8K: by default passes `--gsm-limit 1000` to lm_eval (override with `--gsm-limit 0` for full 1319).

Scheduler mirrors 02_exp_multi_large_pair.py: runs.json, skip completed, poll subprocesses.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Defaults (aligned with 02_exp_multi_large_pair.py)
# ---------------------------------------------------------------------------
GEN_KWARGS = "max_gen_toks=2048,temperature=0,do_sample=False"
PRETRAINED = "Qwen/Qwen2.5-3B-Instruct"
LM_EVAL_MODEL = "seal_jsonl"
BATCH_SIZE = "1"
NUM_FEWSHOT = "0"
# No GPU needed for replay; keep device cpu to avoid loading weights.
DEVICE = "cpu"

# Parallel workers (each runs one lm_eval subprocess)
MAX_PARALLEL = 4


def infer_task(pred_path: Path) -> Optional[str]:
    s = str(pred_path.resolve())
    if "MATH_train" in s or "GSM_train" in s:
        return None
    if f"{os.sep}MATH500{os.sep}" in s:
        return "math_500_cot_zeroshot"
    if f"{os.sep}GSM{os.sep}" in s and "GSM_train" not in s:
        return "gsm8k_cot_zeroshot_unified"
    return None


def collect_jobs(seal_results: Path) -> list[tuple[str, Path, str]]:
    out: list[tuple[str, Path, str]] = []
    for p in sorted(seal_results.rglob("predictions.jsonl")):
        task = infer_task(p)
        if task is None:
            continue
        rel = p.relative_to(seal_results)
        job_id = str(rel).replace(os.sep, "__")
        out.append((job_id, p.resolve(), task))
    return out


def load_runs(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "jobs" in data:
            return data["jobs"]
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return {}


def save_runs(path: Path, jobs: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": datetime.now().isoformat(), "jobs": jobs}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass
class Job:
    job_id: str
    pred_path: Path
    task: str
    outdir: Path
    stdout_log: Path = field(init=False)
    stderr_log: Path = field(init=False)
    proc: Optional[subprocess.Popen] = None
    returncode: Optional[int] = None
    status: str = "pending"
    start_ts: Optional[datetime] = None
    end_ts: Optional[datetime] = None

    def __post_init__(self):
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.stdout_log = self.outdir / "stdout.log"
        self.stderr_log = self.outdir / "stderr.log"

    def build_cmd(self) -> list[str]:
        # pred_path must survive comma-separated model_args — avoid commas in path
        pred_esc = str(self.pred_path)
        model_args = f"pred_path={pred_esc},pretrained={PRETRAINED}"
        return [
            sys.executable,
            "-m",
            "lm_eval",
            "--model",
            LM_EVAL_MODEL,
            "--model_args",
            model_args,
            "--tasks",
            self.task,
            "--device",
            DEVICE,
            "--num_fewshot",
            NUM_FEWSHOT,
            "--batch_size",
            BATCH_SIZE,
            "--gen_kwargs",
            GEN_KWARGS,
            "--apply_chat_template",
            "--output_path",
            str(self.outdir),
            "--log_samples",
        ]

    def cmd_str(self) -> str:
        return shlex.join(self.build_cmd())

    def to_record(self) -> dict:
        duration = None
        if self.start_ts and self.end_ts:
            duration = (self.end_ts - self.start_ts).total_seconds()
        return {
            "job_id": self.job_id,
            "pred_path": str(self.pred_path),
            "task": self.task,
            "status": self.status,
            "returncode": self.returncode,
            "start_ts": self.start_ts.isoformat() if self.start_ts else None,
            "end_ts": self.end_ts.isoformat() if self.end_ts else None,
            "duration_sec": duration,
            "outdir": str(self.outdir),
            "cmd": self.cmd_str(),
        }


def parse_args():
    p = argparse.ArgumentParser(description="SEAL + lm-eval rescoring monitor")
    root = Path(__file__).resolve().parents[1]
    p.add_argument(
        "--seal-results",
        type=Path,
        default=root / "SEAL" / "results",
        help="SEAL results root (contains MATH500/, GSM/, ...)",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=root / "SEAL" / "lm_eval_rescore",
        help="Output root for lm_eval JSON + logs",
    )
    p.add_argument(
        "--runs-json",
        type=Path,
        default=None,
        help="State file (default: <output-root>/runs.json)",
    )
    p.add_argument(
        "--max-parallel",
        type=int,
        default=MAX_PARALLEL,
        help="Max concurrent lm_eval subprocesses",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, append --limit to lm_eval for every task (testing only)",
    )
    p.add_argument(
        "--gsm-limit",
        type=int,
        default=1000,
        help="For gsm8k_cot_zeroshot_unified only: lm_eval --limit (default: 1000). Use 0 for full test set.",
    )
    p.add_argument(
        "--only",
        choices=("all", "gsm", "math500"),
        default="all",
        help="Restrict to GSM8K or MATH-500 rescoring (default: all)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    seal_results = args.seal_results.resolve()
    output_root = args.output_root.resolve()
    runs_path = args.runs_json or (output_root / "runs.json")

    if not seal_results.is_dir():
        print(f"[ERROR] seal-results not found: {seal_results}")
        sys.exit(1)

    # Ensure lm_eval can import seal_jsonl (editable install from harness root)
    harness_root = Path(__file__).resolve().parents[1] / "lm-evaluation-harness"
    if harness_root.is_dir():
        h = str(harness_root)
        if h not in sys.path:
            sys.path.insert(0, h)

    all_specs = collect_jobs(seal_results)
    if args.only == "gsm":
        all_specs = [x for x in all_specs if x[2] == "gsm8k_cot_zeroshot_unified"]
    elif args.only == "math500":
        all_specs = [x for x in all_specs if x[2] == "math_500_cot_zeroshot"]
    # GSM8K first when running both (user preference)
    all_specs.sort(
        key=lambda x: (0 if x[2] == "gsm8k_cot_zeroshot_unified" else 1, x[0])
    )
    if not all_specs:
        print(f"[WARN] No predictions.jsonl found under {seal_results} (MATH500/GSM test only).")
        sys.exit(0)

    runs_state = load_runs(runs_path)
    print(f"[INFO] Found {len(all_specs)} prediction files. State: {runs_path}")

    def effective_job_id(base_id: str, task: str) -> str:
        """Separate cache dirs when GSM uses --gsm-limit (default 1000)."""
        if task == "gsm8k_cot_zeroshot_unified" and args.gsm_limit and args.gsm_limit > 0:
            return f"{base_id}__eval_limit_{args.gsm_limit}"
        return base_id

    queue: list[Job] = []
    for base_job_id, pred_path, task in all_specs:
        job_id = effective_job_id(base_job_id, task)
        outdir = output_root / job_id
        rec = runs_state.get(job_id)
        if rec and rec.get("status") == "done" and rec.get("returncode") == 0:
            print(f"[SKIP] {job_id}")
            continue
        queue.append(Job(job_id=job_id, pred_path=pred_path, task=task, outdir=outdir))

    print(f"[INFO] Queue: {len(queue)} jobs (after skip). Max parallel: {args.max_parallel}")

    running: list[tuple[Job, subprocess.Popen]] = []

    def on_sigint(sig, frame):
        print("\n[CTRL-C] terminating children...")
        for job, proc in running:
            proc.terminate()
        save_runs(runs_path, runs_state)
        sys.exit(130)

    signal.signal(signal.SIGINT, on_sigint)

    def append_limit(cmd: list[str], job: Job) -> list[str]:
        # Global --limit wins for all tasks (debug)
        if args.limit is not None:
            return cmd + ["--limit", str(args.limit)]
        if job.task == "gsm8k_cot_zeroshot_unified" and args.gsm_limit and args.gsm_limit > 0:
            return cmd + ["--limit", str(args.gsm_limit)]
        return cmd

    while queue or running:
        while queue and len(running) < args.max_parallel:
            job = queue.pop(0)
            cmd = append_limit(job.build_cmd(), job)
            job.start_ts = datetime.now()
            job.status = "running"
            stdout_f = open(job.stdout_log, "w", encoding="utf-8")
            stderr_f = open(job.stderr_log, "w", encoding="utf-8")
            print(f"[LAUNCH] {job.job_id} task={job.task}")
            print(f"         {shlex.join(cmd)}")
            env = os.environ.copy()
            hr = Path(__file__).resolve().parents[1] / "lm-evaluation-harness"
            if hr.is_dir():
                prev = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = str(hr) + (os.pathsep + prev if prev else "")
            proc = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f, env=env)
            job.proc = proc
            running.append((job, proc))
            runs_state[job.job_id] = job.to_record()
            save_runs(runs_path, runs_state)

        time.sleep(3)
        still: list[tuple[Job, subprocess.Popen]] = []
        for job, proc in running:
            rc = proc.poll()
            if rc is None:
                still.append((job, proc))
                continue
            job.returncode = rc
            job.end_ts = datetime.now()
            job.status = "done" if rc == 0 else "failed"
            runs_state[job.job_id] = job.to_record()
            save_runs(runs_path, runs_state)
            print(
                f"[DONE] {job.job_id} rc={rc} task={job.task} "
                f"({'OK' if rc == 0 else 'FAIL'})"
            )
        running = still

    print(f"\n[ALL DONE] Summary: {runs_path}")
    # Print metrics from result JSON files if present
    for job_id in sorted(runs_state.keys()):
        rec = runs_state[job_id]
        if rec.get("status") != "done" or rec.get("returncode") != 0:
            continue
        od = Path(rec["outdir"])
        # lm_eval writes results_*json in outdir
        for f in sorted(od.glob("results_*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                res = data.get("results", data)
                print(f"\n=== {job_id} ===")
                print(json.dumps(res, indent=2)[:4000])
            except Exception as e:
                print(f"[WARN] Could not read {f}: {e}")


if __name__ == "__main__":
    main()
