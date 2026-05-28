#!/usr/bin/env python3
"""
Compute per-sample joint metrics for all steering methods:
  - c_i = exp(-NLL_i)                    (compatibility score)
  - J_i = rho_i * exp(-NLL_i)            (DenseCompat)
  - DAS_i = log(rho_i) - NLL_i           (Density-Alignment Score, log version)
  - JointZ = z_rho - z_NLL               (standardized version)

Where:
  NLL_i = mean per-token CE on response-only text (same as compute_ppl_and_rank)
  rho_i = total_tokens / n_steps  (steps split by \\n\\n)

Data sources:
  1. Main grid (FactSteer): eval_grid_less_tokens_3B_lib_manual_same_lib_1000samples/gsm8k_cot_zeroshot
  2. Controls (E0-E3.2): control_experiments/Qwen_Qwen2.5-3B-Instruct/sweep_eval
  3. DenseSteer: exps/gpt_rewrites_unified_new/.../gsm8k_cot_zeroshot_unified_selected_layers
  4. InfamilySteer: exps/large_model_rewrites_unified_new/.../gsm8k_cot_zeroshot_unified_selected_layers

Multi-GPU parallel. Outputs per-(method, layer, lambda) JSON caches + summary + plots.

Usage:
  python new_scripts/joint_metrics_parallel.py --gpus 0,1,2,3,4,5,6,7
  python new_scripts/joint_metrics_parallel.py --plot-only
"""
from __future__ import annotations
import argparse, json, math, multiprocessing as mp, re, statistics, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement")
DOCS = BASE / "documents"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# ── Data source configs ──────────────────────────────────────────────────────

GRID_MAIN = BASE / "lm-evaluation-harness/lm_eval/models/eval_grid_less_tokens_3B_lib_manual_same_lib_1000samples/gsm8k_cot_zeroshot"
GRID_DENSE = BASE / "exps/gpt_rewrites_unified_new/Qwen_Qwen2.5-3B-Instruct/vectors_50_old/Qwen_Qwen2.5-3B-Instruct_applied/gsm8k_cot_zeroshot_unified_selected_layers"
GRID_INFAMILY = BASE / "exps/large_model_rewrites_unified_new/Qwen_Qwen2.5-3B-Instruct/vectors_50_paired_Qwen_Qwen2.5-7B-Instruct/Qwen_Qwen2.5-3B-Instruct_applied/gsm8k_cot_zeroshot_unified_selected_layers"
GRID_CONTROL = BASE / "control_experiments/Qwen_Qwen2.5-3B-Instruct/sweep_eval"

METHODS = {
    "FactSteer": {
        "grid": GRID_MAIN,
        "layers": (17, 31, 35),
        "parse": "main",  # Qwen2.5-3B-Instruct_L{layer}_lam{lam}
    },
    "DenseSteer": {
        "grid": GRID_DENSE,
        "layers": (6, 16, 17, 27, 35),
        "parse": "main",
    },
    "InfamilySteer": {
        "grid": GRID_INFAMILY,
        "layers": (6, 9, 10, 18),
        "parse": "main",
    },
    "E0": {"grid": GRID_CONTROL, "layers": (6,), "parse": "control", "prefix": "e0"},
    "E1": {"grid": GRID_CONTROL, "layers": (6,), "parse": "control", "prefix": "e1"},
    "E2": {"grid": GRID_CONTROL, "layers": (6,), "parse": "control", "prefix": "e2"},
    "E3": {"grid": GRID_CONTROL, "layers": (6,), "parse": "control", "prefix": "e3"},
    "E3.2": {"grid": GRID_CONTROL, "layers": (6,), "parse": "control", "prefix": "e3.2"},
}


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_main_dir(dirname: str) -> tuple[int, float] | None:
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


def parse_control_dir(dirname: str, prefix: str) -> tuple[int, float] | None:
    m = re.match(rf"{re.escape(prefix)}_L(\d+)_lam(.+)$", dirname)
    if not m:
        return None
    layer = int(m.group(1))
    body = m.group(2)
    neg = body.startswith("n")
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


def discover_all_jobs() -> list[dict]:
    """Discover all (method, layer, lambda, samples_path) jobs."""
    jobs = []
    for method, cfg in METHODS.items():
        grid = cfg["grid"]
        if not grid.is_dir():
            print(f"Warning: grid not found for {method}: {grid}", file=sys.stderr)
            continue
        for run_dir in sorted(grid.iterdir()):
            if not run_dir.is_dir():
                continue
            if cfg["parse"] == "main":
                parsed = parse_main_dir(run_dir.name)
            else:
                parsed = parse_control_dir(run_dir.name, cfg["prefix"])
            if parsed is None:
                continue
            layer, lam = parsed
            if layer not in cfg["layers"]:
                continue
            sj = latest_samples_jsonl(run_dir)
            if sj is None or not sj.is_file():
                continue
            jobs.append({
                "method": method,
                "layer": layer,
                "lambda": lam,
                "run_dir": str(run_dir),
                "samples_path": str(sj),
            })
    jobs.sort(key=lambda x: (x["method"], x["layer"], x["lambda"]))
    return jobs


def cache_path(method: str, layer: int, lam: float, cache_root: Path) -> Path:
    lam_s = f"{lam:g}".replace("-", "neg").replace(".", "p")
    return cache_root / f"{method}_L{layer}_lam{lam_s}.json"


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


def compute_rho(text: str, tokenizer) -> float:
    """ρ = total_tokens / n_steps, steps split by \\n\\n."""
    steps = [s.strip() for s in (text or "").split("\n\n") if s.strip()]
    if not steps:
        return 0.0
    n = len(steps)
    total_toks = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in steps)
    return total_toks / n


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
        method = jd["method"]
        layer = int(jd["layer"])
        lam = float(jd["lambda"])
        samples_path = Path(jd["samples_path"])

        lam_s = f"{lam:g}".replace("-", "neg").replace(".", "p")
        cp = cache_dir / f"{method}_L{layer}_lam{lam_s}.json"
        if cp.is_file() and not no_cache:
            print(f"[GPU{gpu_id}] cache hit {cp.name}", flush=True)
            continue

        records = load_records_dedup(samples_path)

        # Per-sample: nll_i, rho_i, ppl_i
        nlls = []      # mean CE per sample (nats)
        rhos = []      # tokens/step per sample
        ppls = []      # exp(mean CE) per sample
        n_skip = 0

        for rec in tqdm(records, desc=f"GPU{gpu_id} {method} L{layer} λ={lam:g}", position=tqdm_pos, leave=True):
            gen = extract_generation(rec)
            if not gen or not gen.strip():
                n_skip += 1
                continue

            # NLL (response-only, same as compute_ppl_and_rank)
            inputs = tokenizer(gen, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
            with th.inference_mode():
                outputs = model(**inputs)
                logits = outputs.logits[:, :-1, :]
                labels = inputs.input_ids[:, 1:]
                loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                nll_i = loss.mean().item()  # mean CE in nats
                ppl_i = th.exp(loss.mean()).item()

            if ma.isnan(nll_i) or ma.isinf(nll_i):
                n_skip += 1
                continue

            rho_i = compute_rho(gen, tokenizer)
            if rho_i <= 0:
                n_skip += 1
                continue

            nlls.append(nll_i)
            rhos.append(rho_i)
            ppls.append(ppl_i)

        N = len(nlls)
        if N == 0:
            blob = {"method": method, "layer": layer, "lambda": lam, "n_docs": len(records), "n_scored": 0, "n_skip": n_skip}
            cp.write_text(jm.dumps(blob, indent=2), encoding="utf-8")
            continue

        # Compute aggregate metrics
        import numpy as _np
        nlls_a = _np.array(nlls)
        rhos_a = _np.array(rhos)

        # c_i = exp(-NLL_i)
        c_vals = _np.exp(-nlls_a)
        # J_i = rho_i * exp(-NLL_i)
        J_vals = rhos_a * c_vals
        # DAS_i = log(rho_i) - NLL_i
        DAS_vals = _np.log(rhos_a) - nlls_a
        # JointZ = z_rho - z_NLL
        mu_rho, sig_rho = float(rhos_a.mean()), float(rhos_a.std(ddof=1)) if N > 1 else 1.0
        mu_nll, sig_nll = float(nlls_a.mean()), float(nlls_a.std(ddof=1)) if N > 1 else 1.0
        z_rho = (rhos_a - mu_rho) / max(sig_rho, 1e-12)
        z_nll = (nlls_a - mu_nll) / max(sig_nll, 1e-12)
        JointZ_vals = z_rho - z_nll

        blob = {
            "method": method, "layer": layer, "lambda": lam,
            "run_dir": str(Path(jd["run_dir"]).relative_to(BASE)),
            "n_docs": len(records), "n_scored": N, "n_skip": n_skip,
            # Aggregate metrics
            "mean_ppl": float(_np.mean(ppls)),
            "mean_nll": float(nlls_a.mean()),
            "mean_rho": float(rhos_a.mean()),
            "mean_c": float(c_vals.mean()),
            "mean_J": float(J_vals.mean()),
            "mean_DAS": float(DAS_vals.mean()),
            "mean_JointZ": float(JointZ_vals.mean()),
            # For standardization reference
            "std_rho": float(rhos_a.std(ddof=1)) if N > 1 else 0.0,
            "std_nll": float(nlls_a.std(ddof=1)) if N > 1 else 0.0,
            "model": cfg_model, "gpu_id": gpu_id,
        }
        cp.write_text(jm.dumps(blob, indent=2), encoding="utf-8")
        print(jm.dumps({
            "gpu": gpu_id, "method": method, "layer": layer, "lambda": lam,
            "ppl": round(blob["mean_ppl"], 4), "rho": round(blob["mean_rho"], 2),
            "J": round(blob["mean_J"], 4), "DAS": round(blob["mean_DAS"], 4),
        }), flush=True)


# ── Plotting ─────────────────────────────────────────────────────────────────

METHOD_STYLES = {
    "FactSteer":      {"color": "#1a237e", "marker": "o"},
    "DenseSteer":     {"color": "#2e7d32", "marker": "s"},
    "InfamilySteer":  {"color": "#e65100", "marker": "^"},
    "E0":             {"color": "#6a1b9a", "marker": "D"},
    "E1":             {"color": "#ad1457", "marker": "v"},
    "E2":             {"color": "#00838f", "marker": "<"},
    "E3":             {"color": "#4e342e", "marker": ">"},
    "E3.2":           {"color": "#546e7a", "marker": "p"},
}

LAYER_ALPHAS = {6: 0.5, 9: 0.55, 10: 0.6, 16: 0.65, 17: 0.8, 18: 0.7, 27: 0.85, 31: 0.9, 35: 1.0}


def load_all_cache(cache_root: Path) -> list[dict]:
    results = []
    for f in sorted(cache_root.glob("*.json")):
        blob = json.loads(f.read_text(encoding="utf-8"))
        if blob.get("n_scored", 0) > 0:
            results.append(blob)
    return results


def write_plots(results: list[dict], out_dir: Path) -> None:
    from matplotlib.ticker import MultipleLocator

    metrics = [
        ("mean_ppl", "Perplexity (PPL)", "PPL"),
        ("mean_rho", r"Reasoning Density ($\rho$)", "rho"),
        ("mean_c", r"Compatibility $c = \exp(-\mathrm{NLL})$", "compatibility"),
        ("mean_J", r"DenseCompat $J = \rho \cdot \exp(-\mathrm{NLL})$", "DenseCompat"),
        ("mean_DAS", r"DAS $= \log\rho - \mathrm{NLL}$", "DAS"),
        ("mean_JointZ", r"JointZ $= z_\rho - z_{\mathrm{NLL}}$", "JointZ"),
    ]

    # Group by method
    by_method: dict[str, list[dict]] = {}
    for r in results:
        by_method.setdefault(r["method"], []).append(r)

    # ── Per-metric, all methods on one plot (best layer per method) ──
    methods_all = [
        "FactSteer", "DenseSteer", "InfamilySteer", "E0", "E1", "E2", "E3", "E3.2",
    ]
    for metric_key, ylabel, fname in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        methods = [m for m in methods_all if not (fname == "DAS" and m == "FactSteer")]
        for method in methods:
            recs = by_method.get(method, [])
            if not recs:
                continue
            style = METHOD_STYLES.get(method, {"color": "#333", "marker": "o"})
            # Group by layer
            by_layer: dict[int, list] = {}
            for r in recs:
                by_layer.setdefault(r["layer"], []).append(r)
            # Pick layer with widest lambda range
            best_layer = max(by_layer.keys(), key=lambda L: len(by_layer[L]))
            recs_l = sorted(by_layer[best_layer], key=lambda r: r["lambda"])
            xs = [r["lambda"] for r in recs_l]
            ys = [r[metric_key] for r in recs_l]
            ax.plot(xs, ys, marker=style["marker"], linewidth=1.5, markersize=4,
                    color=style["color"], label=f"{method} (L{best_layer})")
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7, ncol=2)
        ax.set_title(f"{ylabel} vs Steering Strength")
        if fname == "DAS":
            ax.set_xlim(-7.5, 7.5)
        fig.tight_layout()
        outp = out_dir / f"joint_{fname}_vs_lambda.png"
        fig.savefig(outp, dpi=200)
        plt.close()
        print(f"wrote {outp}")

    # ── Per-method multi-layer plots for key metrics ──
    key_metrics = [
        ("mean_J", r"DenseCompat $J$", "DenseCompat"),
        ("mean_DAS", r"DAS", "DAS"),
    ]
    for method, recs in sorted(by_method.items()):
        by_layer: dict[int, list] = {}
        for r in recs:
            by_layer.setdefault(r["layer"], []).append(r)
        if len(by_layer) <= 1:
            continue
        for metric_key, ylabel, fname in key_metrics:
            fig, ax = plt.subplots(figsize=(9, 5.5))
            style = METHOD_STYLES.get(method, {"color": "#333", "marker": "o"})
            for L in sorted(by_layer.keys()):
                recs_l = sorted(by_layer[L], key=lambda r: r["lambda"])
                xs = [r["lambda"] for r in recs_l]
                ys = [r[metric_key] for r in recs_l]
                alpha = LAYER_ALPHAS.get(L, 0.7)
                ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3,
                        alpha=alpha, label=f"L{L}")
            ax.set_xlabel(r"$\lambda$")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)
            ax.set_title(f"{method}: {ylabel} vs λ")
            fig.tight_layout()
            outp = out_dir / f"joint_{fname}_{method}_multilayer.png"
            fig.savefig(outp, dpi=200)
            plt.close()
            print(f"wrote {outp}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    cache_root = DOCS / "joint_metrics_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        results = load_all_cache(cache_root)
        print(f"Loaded {len(results)} cached results")
        out_json = DOCS / "joint_metrics_grid_eval.json"
        out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        if not args.skip_plots:
            write_plots(results, DOCS)
        return

    jobs = discover_all_jobs()
    print(f"Discovered {len(jobs)} total jobs")
    for method in sorted(set(j["method"] for j in jobs)):
        n = sum(1 for j in jobs if j["method"] == method)
        print(f"  {method}: {n} jobs")

    # Filter cached
    pending = []
    for jd in jobs:
        cp = cache_path(jd["method"], jd["layer"], jd["lambda"], cache_root)
        if cp.is_file() and not args.no_cache:
            continue
        pending.append(jd)
    print(f"Pending (not cached): {len(pending)}")

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
                "max_length": args.max_length, "no_cache": args.no_cache, "dtype": args.dtype,
            }
            p = ctx.Process(target=gpu_worker, args=(payload,))
            p.start(); procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                print(f"Worker exited {p.exitcode}", file=sys.stderr)
                sys.exit(p.exitcode if p.exitcode is not None else 1)

    results = load_all_cache(cache_root)
    out_json = DOCS / "joint_metrics_grid_eval.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_json} ({len(results)} entries)")

    if not args.skip_plots:
        write_plots(results, DOCS)


if __name__ == "__main__":
    if mp.parent_process() is None:
        main()
