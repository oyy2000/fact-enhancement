#!/usr/bin/env python3
"""
Compute response-only PPL (compute_ppl_and_rank style) for control experiments E0–E3.2.

Each experiment has sweep_eval/{exp}_L6_lam{...} directories with lm_eval samples.
Lambda range: [-5, 5] step 0.5 (21 configs per experiment).

Multi-GPU: each GPU loads a full model copy; (experiment, λ) jobs are round-robin sharded.

Usage:
  /common/users/sl2148/anaconda3/envs/fact_yang/bin/python new_scripts/control_ppl_parallel.py --gpus 0,1,2,3,4,5,6,7
  /common/users/sl2148/anaconda3/envs/fact_yang/bin/python new_scripts/control_ppl_parallel.py --plot-only
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import re
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement")
SWEEP_DIR = BASE / "control_experiments" / "Qwen_Qwen2.5-3B-Instruct" / "sweep_eval"
DOCS = BASE / "documents"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

EXPERIMENTS = ("e0", "e1", "e2", "e3", "e3.2")
LAYER = 6  # all controls use L6


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_sweep_dir(dirname: str) -> tuple[str, float] | None:
    """Parse e0_L6_lam5p0 or e0_L6_lamn1p5 → ('e0', 5.0) or ('e0', -1.5)."""
    m = re.match(r"(e\d+(?:\.\d+)?)_L(\d+)_lam(.+)$", dirname)
    if not m:
        return None
    exp = m.group(1)
    body = m.group(3)
    neg = body.startswith("n")
    if neg:
        body = body[1:]
    val = float(body.replace("p", "."))
    return exp, -val if neg else val


def latest_samples_jsonl(run_dir: Path) -> Path | None:
    subdirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("Qwen__")]
    if not subdirs:
        return None
    search = subdirs[0]
    cands = sorted(search.glob("samples_*.jsonl"))
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def discover_jobs(
    experiments: tuple[str, ...],
    target_lambdas: list[float],
) -> list[dict]:
    jobs = []
    for run_dir in sorted(SWEEP_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = parse_sweep_dir(run_dir.name)
        if parsed is None:
            continue
        exp, lam = parsed
        if exp not in experiments:
            continue
        if not any(math.isclose(lam, t, abs_tol=1e-5) for t in target_lambdas):
            continue
        sj = latest_samples_jsonl(run_dir)
        if sj is None or not sj.is_file():
            continue
        jobs.append({
            "experiment": exp,
            "lambda": lam,
            "run_dir": str(run_dir),
            "samples_path": str(sj),
        })
    jobs.sort(key=lambda x: (x["experiment"], x["lambda"]))
    return jobs


def cache_path(exp: str, lam: float, cache_root: Path) -> Path:
    lam_s = f"{lam:g}".replace("-", "neg").replace(".", "p")
    return cache_root / f"{exp}_lam{lam_s}.json"


def load_records_dedup(path: Path) -> list[dict]:
    """Load jsonl, deduplicate by doc_id."""
    seen = set()
    records = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            did = rec.get("doc_id")
            if did in seen:
                continue
            seen.add(did)
            records.append(rec)
    return records


def extract_generation(rec: dict) -> str:
    resps = rec.get("resps", [[]])
    if resps and resps[0]:
        return resps[0][0]
    return ""


# ── GPU worker ───────────────────────────────────────────────────────────────

def gpu_worker(payload: dict) -> None:
    import os
    gpu_id = int(payload["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import json as jm
    import math as ma
    import torch as th
    from torch.nn import CrossEntropyLoss as CEL
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM as AM, AutoTokenizer as AT

    cfg_model = payload["model"]
    max_length = int(payload["max_length"])
    max_docs = payload.get("max_docs")
    no_cache = bool(payload["no_cache"])
    cache_dir = Path(payload["cache_dir"])
    tqdm_pos = int(payload.get("tqdm_position", gpu_id % 8))
    dtype_str = payload["dtype"]

    dt = th.float16 if dtype_str == "float16" else th.bfloat16
    tokenizer = AT.from_pretrained(cfg_model, trust_remote_code=True)
    model = AM.from_pretrained(cfg_model, torch_dtype=dt, device_map={"": "cuda:0"}, trust_remote_code=True)
    model.eval()
    loss_fct = CEL(reduction="none")

    cache_dir.mkdir(parents=True, exist_ok=True)

    for jd in payload["jobs"]:
        exp = jd["experiment"]
        lam = float(jd["lambda"])
        samples_path = Path(jd["samples_path"])

        lam_s = f"{lam:g}".replace("-", "neg").replace(".", "p")
        cp = cache_dir / f"{exp}_lam{lam_s}.json"
        if cp.is_file() and not no_cache:
            print(f"[GPU{gpu_id}] cache hit {cp.name}", flush=True)
            continue

        records = load_records_dedup(samples_path)
        if max_docs is not None:
            records = records[:int(max_docs)]

        ppl_vals = []
        n_skip = 0

        for rec in tqdm(records, desc=f"GPU{gpu_id} {exp} λ={lam:g}", position=tqdm_pos, leave=True):
            gen = extract_generation(rec)
            if not gen or not gen.strip():
                n_skip += 1
                continue

            inputs = tokenizer(gen, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
            with th.inference_mode():
                outputs = model(**inputs)
                logits = outputs.logits[:, :-1, :]
                labels = inputs.input_ids[:, 1:]
                sl = logits.reshape(-1, logits.size(-1))
                sla = labels.reshape(-1)
                loss = loss_fct(sl, sla)
                ppl_val = th.exp(loss.mean()).item()

            if ma.isnan(ppl_val) or ma.isinf(ppl_val):
                n_skip += 1
            else:
                ppl_vals.append(ppl_val)

        ppl_mean = (sum(ppl_vals) / len(ppl_vals)) if ppl_vals else None

        blob = {
            "experiment": exp,
            "lambda": lam,
            "layer": LAYER,
            "run_dir": str(Path(jd["run_dir"]).relative_to(BASE)),
            "samples_jsonl": str(samples_path.relative_to(BASE)),
            "n_docs": len(records),
            "ppl_mean": ppl_mean,
            "n_scored": len(ppl_vals),
            "n_skip": n_skip,
            "max_length": max_length,
            "model": cfg_model,
            "gpu_id": gpu_id,
        }
        cp.write_text(jm.dumps(blob, indent=2), encoding="utf-8")
        print(jm.dumps({"gpu": gpu_id, "exp": exp, "lambda": lam, "ppl_mean": ppl_mean}), flush=True)


# ── plotting ─────────────────────────────────────────────────────────────────

EXP_COLORS = {
    "e0": "#1a237e",
    "e1": "#3949ab",
    "e2": "#5c6bc0",
    "e3": "#7986cb",
    "e3.2": "#9fa8da",
}
EXP_LABELS = {
    "e0": "E0 (fact-enhanced)",
    "e1": "E1 (random-token control)",
    "e2": "E2 (shuffled-fact control)",
    "e3": "E3 (reversed-fact control)",
    "e3.2": "E3.2 (paraphrase control)",
}


def write_plots(
    series: dict[str, dict[float, float]],
    lam_sorted: list[float],
    out_dir: Path,
) -> None:
    from matplotlib.ticker import MultipleLocator

    plt.figure(figsize=(9, 5.5))
    for exp in EXPERIMENTS:
        if exp not in series:
            continue
        ys = [series[exp].get(l, float("nan")) for l in lam_sorted]
        plt.plot(
            lam_sorted, ys,
            marker="o", linewidth=1.5, markersize=4,
            color=EXP_COLORS.get(exp, "#333"),
            label=EXP_LABELS.get(exp, exp),
        )
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Perplexity (PPL)")
    plt.xlim(min(lam_sorted) - 0.3, max(lam_sorted) + 0.3)
    plt.gca().xaxis.set_major_locator(MultipleLocator(1.0))
    plt.grid(True, alpha=0.35)
    plt.legend(loc="best", fontsize=8)
    plt.title("Impact of Steering Strength on Token-Level PPL (Control Experiments, L6)")
    plt.figtext(
        0.5, 0.02,
        "Response-only PPL = exp(mean CE over all response tokens). "
        "Same as ModelEvaluator.compute_ppl_and_rank on generated_text. "
        "Qwen2.5-3B-Instruct base model, GSM8K CoT.",
        ha="center", fontsize=7,
    )
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    outp = out_dir / "control_ppl_vs_lambda_L6.png"
    plt.savefig(outp, dpi=200)
    plt.close()
    print(f"wrote {outp}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    cache_root = DOCS / "control_ppl_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    target_lambdas = [round(-5.0 + i * 0.5, 1) for i in range(21)]  # -5.0 to +5.0

    # Load from cache
    series: dict[str, dict[float, float]] = {exp: {} for exp in EXPERIMENTS}
    for exp in EXPERIMENTS:
        for lam in target_lambdas:
            cp = cache_path(exp, lam, cache_root)
            if cp.is_file():
                blob = json.loads(cp.read_text(encoding="utf-8"))
                v = blob.get("ppl_mean")
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    series[exp][lam] = float(v)

    if args.plot_only:
        out_json = DOCS / "control_ppl_grid_eval.json"
        out_json.write_text(json.dumps(series, indent=2), encoding="utf-8")
        if not args.skip_plots:
            write_plots(series, target_lambdas, DOCS)
        return

    jobs = discover_jobs(EXPERIMENTS, target_lambdas)
    expect_n = len(EXPERIMENTS) * len(target_lambdas)
    if len(jobs) != expect_n:
        print(f"Warning: expected {expect_n} jobs, found {len(jobs)}", file=sys.stderr)

    # Filter out already-cached
    pending = []
    for jd in jobs:
        cp = cache_path(jd["experiment"], jd["lambda"], cache_root)
        if cp.is_file() and not args.no_cache:
            print(f"cache hit {cp.name}")
            continue
        pending.append(jd)

    if pending:
        gpu_list = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
        if not gpu_list:
            print("Empty --gpus", file=sys.stderr)
            sys.exit(1)

        buckets: list[list[dict]] = [[] for _ in gpu_list]
        for i, jd in enumerate(pending):
            buckets[i % len(gpu_list)].append(jd)

        ctx = mp.get_context("spawn")
        procs = []
        for wi, gid in enumerate(gpu_list):
            sub = buckets[wi]
            if not sub:
                continue
            payload = {
                "gpu_id": gid,
                "tqdm_position": wi,
                "jobs": sub,
                "cache_dir": str(cache_root),
                "model": args.model,
                "max_length": args.max_length,
                "max_docs": args.max_docs,
                "no_cache": args.no_cache,
                "dtype": args.dtype,
            }
            p = ctx.Process(target=gpu_worker, args=(payload,))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
            if p.exitcode != 0:
                print(f"Worker exited with code {p.exitcode}", file=sys.stderr)
                sys.exit(p.exitcode if p.exitcode is not None else 1)

    # Reload all cache
    for exp in EXPERIMENTS:
        for lam in target_lambdas:
            cp = cache_path(exp, lam, cache_root)
            if cp.is_file():
                blob = json.loads(cp.read_text(encoding="utf-8"))
                v = blob.get("ppl_mean")
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    series[exp][lam] = float(v)

    out_json = DOCS / "control_ppl_grid_eval.json"
    out_json.write_text(json.dumps(series, indent=2), encoding="utf-8")
    print(f"wrote {out_json}")

    if not args.skip_plots:
        write_plots(series, target_lambdas, DOCS)


if __name__ == "__main__":
    if mp.parent_process() is None:
        main()
