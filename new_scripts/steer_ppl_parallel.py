#!/usr/bin/env python3
"""
Compute response-only PPL for DenseSteer (gpt_rewrites) and InfamilySteer (large_model_rewrites).

Multi-GPU parallel, same compute_ppl_and_rank style as control_ppl_parallel.py.

Usage:
  python new_scripts/steer_ppl_parallel.py --steer dense --gpus 0,1,2,3,4,5,6,7
  python new_scripts/steer_ppl_parallel.py --steer infamily --gpus 0,1,2,3,4,5,6,7
  python new_scripts/steer_ppl_parallel.py --steer dense --plot-only
  python new_scripts/steer_ppl_parallel.py --steer infamily --plot-only
"""
from __future__ import annotations
import argparse, json, math, multiprocessing as mp, re, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement")
DOCS = BASE / "documents"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

STEER_CFG = {
    "dense": {
        "grid": BASE / "exps/gpt_rewrites_unified_new/Qwen_Qwen2.5-3B-Instruct/vectors_50_old/Qwen_Qwen2.5-3B-Instruct_applied/gsm8k_cot_zeroshot_unified_selected_layers",
        "layers": (6, 16, 17, 27, 35),
        "lambdas": sorted(set(
            [round(x, 1) for x in [i * 2.0 for i in range(-7, 8)] if abs(x) <= 14]
        )),  # -14,-12,...,0,...,12,14 step 2
        "cache_tag": "dense_ppl_cache",
        "plot_title": "DenseSteer (GPT rewrites)",
        "plot_stem": "dense_ppl",
    },
    "infamily": {
        "grid": BASE / "exps/large_model_rewrites_unified_new/Qwen_Qwen2.5-3B-Instruct/vectors_50_paired_Qwen_Qwen2.5-7B-Instruct/Qwen_Qwen2.5-3B-Instruct_applied/gsm8k_cot_zeroshot_unified_selected_layers",
        "layers": (6, 9, 10, 18),
        "lambdas": sorted(set(
            [round(x, 2) for x in [i * 0.05 for i in range(-10, 11)] if abs(x) <= 0.5]
        )),  # -0.5,-0.45,...,0,...,0.45,0.5 step 0.05
        "cache_tag": "infamily_ppl_cache",
        "plot_title": "InfamilySteer (7B→3B)",
        "plot_stem": "infamily_ppl",
    },
}


def parse_layer_lambda(dirname: str) -> tuple[int, float] | None:
    m = re.match(r"Qwen2\.5-3B-Instruct_L(\d+)_lam(.+)$", dirname)
    if not m:
        return None
    layer = int(m.group(1))
    body = m.group(2)
    neg = body.startswith("-")
    if neg:
        body = body[1:]
    val = float(body.replace("p", "."))
    return layer, -val if neg else val


def latest_samples_jsonl(run_dir: Path) -> Path | None:
    subdirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("Qwen__")]
    if not subdirs:
        return None
    cands = sorted(subdirs[0].glob("samples_*.jsonl"))
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def discover_jobs(grid: Path, layers: tuple, lambdas: list[float]) -> list[dict]:
    jobs = []
    for run_dir in sorted(grid.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = parse_layer_lambda(run_dir.name)
        if parsed is None:
            continue
        layer, lam = parsed
        if layer not in layers:
            continue
        if not any(math.isclose(lam, t, abs_tol=1e-5) for t in lambdas):
            continue
        sj = latest_samples_jsonl(run_dir)
        if sj is None or not sj.is_file():
            continue
        jobs.append({"layer": layer, "lambda": lam, "run_dir": str(run_dir), "samples_path": str(sj)})
    jobs.sort(key=lambda x: (x["layer"], x["lambda"]))
    return jobs


def cache_path(layer: int, lam: float, cache_root: Path) -> Path:
    lam_s = f"{lam:g}".replace("-", "neg").replace(".", "p")
    return cache_root / f"L{layer}_lam{lam_s}.json"


def load_records_dedup(path: Path) -> list[dict]:
    seen = set(); records = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            did = rec.get("doc_id")
            if did in seen: continue
            seen.add(did); records.append(rec)
    return records


def extract_generation(rec: dict) -> str:
    resps = rec.get("resps", [[]])
    return resps[0][0] if resps and resps[0] else ""


# ── GPU worker ───────────────────────────────────────────────────────────────

def gpu_worker(payload: dict) -> None:
    import os
    gpu_id = int(payload["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import json as jm, math as ma, torch as th
    from torch.nn import CrossEntropyLoss as CEL
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM as AM, AutoTokenizer as AT

    cfg_model = payload["model"]
    max_length = int(payload["max_length"])
    max_docs = payload.get("max_docs")
    no_cache = bool(payload["no_cache"])
    cache_dir = Path(payload["cache_dir"])
    tqdm_pos = int(payload.get("tqdm_position", gpu_id % 8))
    dt = th.float16 if payload["dtype"] == "float16" else th.bfloat16

    tokenizer = AT.from_pretrained(cfg_model, trust_remote_code=True)
    model = AM.from_pretrained(cfg_model, torch_dtype=dt, device_map={"": "cuda:0"}, trust_remote_code=True)
    model.eval()
    loss_fct = CEL(reduction="none")
    cache_dir.mkdir(parents=True, exist_ok=True)

    for jd in payload["jobs"]:
        layer = int(jd["layer"])
        lam = float(jd["lambda"])
        samples_path = Path(jd["samples_path"])

        lam_s = f"{lam:g}".replace("-", "neg").replace(".", "p")
        cp = cache_dir / f"L{layer}_lam{lam_s}.json"
        if cp.is_file() and not no_cache:
            print(f"[GPU{gpu_id}] cache hit {cp.name}", flush=True)
            continue

        records = load_records_dedup(samples_path)
        if max_docs is not None:
            records = records[:int(max_docs)]

        ppl_vals = []
        n_skip = 0
        for rec in tqdm(records, desc=f"GPU{gpu_id} L{layer} λ={lam:g}", position=tqdm_pos, leave=True):
            gen = extract_generation(rec)
            if not gen or not gen.strip():
                n_skip += 1; continue
            inputs = tokenizer(gen, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
            with th.inference_mode():
                outputs = model(**inputs)
                logits = outputs.logits[:, :-1, :]
                labels = inputs.input_ids[:, 1:]
                loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                ppl_val = th.exp(loss.mean()).item()
            if ma.isnan(ppl_val) or ma.isinf(ppl_val):
                n_skip += 1
            else:
                ppl_vals.append(ppl_val)

        ppl_mean = (sum(ppl_vals) / len(ppl_vals)) if ppl_vals else None
        blob = {
            "layer": layer, "lambda": lam,
            "run_dir": str(Path(jd["run_dir"]).relative_to(BASE)),
            "samples_jsonl": str(samples_path.relative_to(BASE)),
            "n_docs": len(records), "ppl_mean": ppl_mean,
            "n_scored": len(ppl_vals), "n_skip": n_skip,
            "max_length": max_length, "model": cfg_model, "gpu_id": gpu_id,
        }
        cp.write_text(jm.dumps(blob, indent=2), encoding="utf-8")
        print(jm.dumps({"gpu": gpu_id, "layer": layer, "lambda": lam, "ppl_mean": ppl_mean}), flush=True)


# ── plotting ─────────────────────────────────────────────────────────────────

LAYER_COLORS = {
    6: "#1a237e", 9: "#283593", 10: "#3949ab",
    16: "#5c6bc0", 17: "#7986cb", 18: "#9fa8da",
    27: "#c5cae9", 35: "#e8eaf6",
}


def write_plots(series, lam_sorted, layers, out_dir, plot_stem, title):
    from matplotlib.ticker import MultipleLocator

    plt.figure(figsize=(9, 5.5))
    for L in sorted(layers):
        if L not in series:
            continue
        ys = [series[L].get(l, float("nan")) for l in lam_sorted]
        plt.plot(lam_sorted, ys, marker="o", linewidth=1.5, markersize=4,
                 color=LAYER_COLORS.get(L, "#333"),
                 label=f"Qwen2.5-3B-Instruct-L{L}")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Perplexity (PPL)")
    span = max(lam_sorted) - min(lam_sorted)
    plt.xlim(min(lam_sorted) - span * 0.03, max(lam_sorted) + span * 0.03)
    tick = 2.0 if span > 10 else (0.1 if span <= 1 else 1.0)
    plt.gca().xaxis.set_major_locator(MultipleLocator(tick))
    plt.grid(True, alpha=0.35)
    plt.legend(loc="best", fontsize=8)
    plt.title(f"Impact of Steering Strength on Token-Level PPL ({title})")
    plt.figtext(0.5, 0.02,
        "Response-only PPL = exp(mean CE). Same as ModelEvaluator.compute_ppl_and_rank. "
        "Qwen2.5-3B-Instruct base, GSM8K CoT.",
        ha="center", fontsize=7)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    outp = out_dir / f"{plot_stem}_vs_lambda.png"
    plt.savefig(outp, dpi=200)
    plt.close()
    print(f"wrote {outp}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steer", required=True, choices=("dense", "infamily"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    cfg = STEER_CFG[args.steer]
    grid = cfg["grid"]
    layers = cfg["layers"]
    lambdas = cfg["lambdas"]
    cache_root = DOCS / cfg["cache_tag"]
    cache_root.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    # Load cache
    series = {L: {} for L in layers}
    for L in layers:
        for lam in lambdas:
            cp = cache_path(L, lam, cache_root)
            if cp.is_file():
                blob = json.loads(cp.read_text(encoding="utf-8"))
                v = blob.get("ppl_mean")
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    series[L][lam] = float(v)

    if args.plot_only:
        out_json = DOCS / f"{cfg['plot_stem']}_grid_eval.json"
        out_json.write_text(json.dumps({str(k): v for k, v in series.items()}, indent=2), encoding="utf-8")
        if not args.skip_plots:
            write_plots(series, lambdas, layers, DOCS, cfg["plot_stem"], cfg["plot_title"])
        return

    jobs = discover_jobs(grid, layers, lambdas)
    expect = len(layers) * len(lambdas)
    if len(jobs) != expect:
        print(f"Warning: expected {expect} jobs, found {len(jobs)}", file=sys.stderr)

    pending = []
    for jd in jobs:
        cp = cache_path(jd["layer"], jd["lambda"], cache_root)
        if cp.is_file() and not args.no_cache:
            print(f"cache hit {cp.name}")
            continue
        pending.append(jd)

    if pending:
        gpu_list = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
        buckets = [[] for _ in gpu_list]
        for i, jd in enumerate(pending):
            buckets[i % len(gpu_list)].append(jd)
        ctx = mp.get_context("spawn")
        procs = []
        for wi, gid in enumerate(gpu_list):
            sub = buckets[wi]
            if not sub: continue
            payload = {
                "gpu_id": gid, "tqdm_position": wi, "jobs": sub,
                "cache_dir": str(cache_root), "model": args.model,
                "max_length": args.max_length, "max_docs": args.max_docs,
                "no_cache": args.no_cache, "dtype": args.dtype,
            }
            p = ctx.Process(target=gpu_worker, args=(payload,))
            p.start(); procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                print(f"Worker exited {p.exitcode}", file=sys.stderr)
                sys.exit(p.exitcode if p.exitcode is not None else 1)

    # Reload cache
    for L in layers:
        for lam in lambdas:
            cp = cache_path(L, lam, cache_root)
            if cp.is_file():
                blob = json.loads(cp.read_text(encoding="utf-8"))
                v = blob.get("ppl_mean")
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    series[L][lam] = float(v)

    out_json = DOCS / f"{cfg['plot_stem']}_grid_eval.json"
    out_json.write_text(json.dumps({str(k): v for k, v in series.items()}, indent=2), encoding="utf-8")
    print(f"wrote {out_json}")

    if not args.skip_plots:
        write_plots(series, lambdas, layers, DOCS, cfg["plot_stem"], cfg["plot_title"])


if __name__ == "__main__":
    if mp.parent_process() is None:
        main()
