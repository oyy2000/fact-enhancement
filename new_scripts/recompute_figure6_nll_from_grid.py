#!/usr/bin/env python3
"""
Recompute token-level NLL on GSM8K CoT samples from the eval grid (steered generations),
using two tokenization / scoring conventions:

  (1) control_generation_nll.py: single-string tokenization of prompt+continuation with
      prompt masked via -100 (HF loss on generated tokens only).

  (2) 03_PRM_plots.py / 06_PRM_plots_2.py (ModelEvaluator.compute_rank_shift): prompt
      tokenized with add_special_tokens=True, continuation with add_special_tokens=False,
      concatenated; mean cross-entropy on continuation positions only.

  (3) 06_PRM_plots_2.py (ModelEvaluator.compute_ppl_and_rank): full-text PPL.
      tokenizer(generated_text) → CE over ALL tokens → exp(mean_loss).
      This is what produces the ``ppl`` field in prm_results/results_merged.json
      and the ~1.7 y-axis in the original Figure 6.

Produces Figure-6-style line plots (λ vs mean NLL) for layers L17, L31, L35 and writes
outputs under documents/.

Run with the project env that has torch + transformers + matplotlib, e.g.:
  /common/users/sl2148/anaconda3/envs/fact_yang/bin/python new_scripts/recompute_figure6_nll_from_grid.py
  /common/users/sl2148/anaconda3/envs/fact_yang/bin/python new_scripts/recompute_figure6_nll_from_grid.py --gpus 0,1,2,3,4,5,6,7

Multi-GPU: each GPU runs one process, full model on that card (``device_map`` to a single device),
jobs are round-robin sharded. Single-GPU: ``--gpus 0`` (no process pool).

Metrics are mean cross-entropy (natural log, nats) over generated tokens, corpus-weighted
by token count (same as control_generation_nll ``weighted_mean_nll``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path("/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement")
GRID = (
    BASE
    / "lm-evaluation-harness/lm_eval/models/eval_grid_less_tokens_3B_lib_manual_same_lib_1000samples/gsm8k_cot_zeroshot"
)
DOCS = BASE / "documents"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

DEFAULT_TARGET_LAYERS = (17, 31, 35)
# Original coarse grid (2.5 step on [-10,10])
DEFAULT_TARGET_LAMBDAS = [-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0]


def build_lambda_grid(lam_min: float, lam_max: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("lambda step must be positive")
    n = int(round((lam_max - lam_min) / step))
    if n < 0:
        raise ValueError("lambda_max < lambda_min")
    out: list[float] = []
    for i in range(n + 1):
        lam = lam_min + i * step
        out.append(round(lam + 0.0, 10))  # stabilize -10.0 vs -9.999999
    return out


def _load_control_generation_nll():
    path = BASE / "new_scripts/my_scripts/control_generation_nll.py"
    spec = importlib.util.spec_from_file_location("control_generation_nll", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["control_generation_nll"] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_layer_lambda(dirname: str) -> tuple[int, float] | None:
    """Parse Qwen2.5-3B-Instruct_L35_lam-10p0 or ..._L35_BASELINE -> (35, -10.0) or (35, 0.0)."""
    m = re.match(r"Qwen2\.5-3B-Instruct_(L\d+)_(.+)$", dirname)
    if not m:
        return None
    layer = int(m.group(1)[1:])
    rest = m.group(2)
    if rest == "BASELINE":
        return layer, 0.0
    if not rest.startswith("lam"):
        return None
    body = rest[3:]  # e.g. -10p0, 7p5
    neg = body.startswith("-")
    if neg:
        body = body[1:]
    val = float(body.replace("p", "."))
    return layer, -val if neg else val


def latest_samples_jsonl(run_dir: Path) -> Path | None:
    subdirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("Qwen__")]
    if not subdirs:
        return None
    search = subdirs[0]
    cands = sorted(search.glob("samples_*.jsonl"))
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def nll_prm_concat_style(
    prompt: str,
    continuation: str,
    model,
    tokenizer,
    max_length: int,
) -> tuple[float | None, int]:
    """Match 03_PRM_plots / 06_PRM_plots_2 rank-shift tokenization; mean CE on continuation."""
    if not continuation.strip():
        return None, 0

    prompt_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )
    response_inputs = tokenizer(
        continuation,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    prompt_ids = prompt_inputs.input_ids.to(model.device)
    response_ids = response_inputs.input_ids.to(model.device)
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    if input_ids.shape[1] > max_length:
        return None, 0

    start_idx = int(prompt_ids.shape[1])
    if start_idx == 0 or input_ids.shape[1] <= start_idx:
        return None, 0

    loss_fct = CrossEntropyLoss(reduction="mean")
    with torch.inference_mode():
        out = model(input_ids)
        logits = out.logits
    shift_logits = logits[:, start_idx - 1 : -1, :]
    shift_labels = input_ids[:, start_idx:]
    if shift_logits.shape[1] != shift_labels.shape[1]:
        return None, int(shift_labels.numel())

    loss = loss_fct(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
    )
    if torch.isnan(loss) or torch.isinf(loss):
        return None, int(shift_labels.numel())
    return float(loss.item()), int(shift_labels.numel())


def ppl_full_text(
    text: str,
    model,
    tokenizer,
    max_length: int,
) -> float | None:
    """
    Exact replica of ModelEvaluator.compute_ppl_and_rank (PPL part only).
    tokenizer(text) → CE(reduction=none) over ALL tokens → exp(mean).
    Returns PPL (not NLL).
    NOTE: reference prm_results uses generated_text = response only (no prompt).
    """
    if not text or not text.strip():
        return None
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :]
        labels = inputs.input_ids[:, 1:]
        shift_logits = logits.reshape(-1, logits.size(-1))
        shift_labels = labels.reshape(-1)
        loss_fct = CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits, shift_labels)
        ppl = torch.exp(loss.mean()).item()
    if math.isnan(ppl) or math.isinf(ppl):
        return None
    return ppl


def discover_jobs(
    target_lambdas: list[float],
    target_layers: tuple[int, ...] = DEFAULT_TARGET_LAYERS,
    grid: Path = GRID,
) -> list[tuple[int, float, Path, Path]]:
    jobs: list[tuple[int, float, Path, Path]] = []
    if not grid.is_dir():
        raise FileNotFoundError(grid)
    for run_dir in sorted(grid.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = parse_layer_lambda(run_dir.name)
        if parsed is None:
            continue
        layer, lam = parsed
        if layer not in target_layers:
            continue
        if not any(math.isclose(lam, t, abs_tol=1e-5) for t in target_lambdas):
            continue
        sj = latest_samples_jsonl(run_dir)
        if sj is None or not sj.is_file():
            continue
        jobs.append((layer, lam, run_dir, sj))
    jobs.sort(key=lambda x: (x[0], x[1]))
    return jobs


def cache_path(layer: int, lam: float, cache_root: Path) -> Path:
    lam_s = f"{lam:g}".replace("-", "neg").replace(".", "p")
    return cache_root / f"L{layer}_lam{lam_s}.json"


def load_series_from_cache(
    cache_root: Path,
    target_lambdas: list[float],
    target_layers: tuple[int, ...],
    prm_only: bool,
    ppl_mode: bool = False,
) -> dict[str, dict[int, dict[float, float]]]:
    if ppl_mode:
        keys = ["ppl_full_text"]
    elif prm_only:
        keys = ["prm_prompt_resp_concat"]
    else:
        keys = ["control_labels_mask", "prm_prompt_resp_concat"]
    series: dict[str, dict[int, dict[float, float]]] = {k: {L: {} for L in target_layers} for k in keys}
    for layer in target_layers:
        for lam in target_lambdas:
            cp = cache_path(layer, lam, cache_root)
            if not cp.is_file():
                continue
            blob = json.loads(cp.read_text(encoding="utf-8"))
            for key in keys:
                v = blob.get(key)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    series[key][layer][lam] = float(v)
    return series


def _job_to_dict(layer: int, lam: float, run_dir: Path, samples_path: Path) -> dict:
    return {
        "layer": layer,
        "lambda": lam,
        "run_dir": str(run_dir),
        "samples_path": str(samples_path),
    }


def _weighted_mean(ns: list[float], ts: list[int]) -> float | None:
    if not ns or not ts or sum(ts) == 0:
        return None
    return sum(n * t for n, t in zip(ns, ts)) / sum(ts)


def _gpu_worker_run(payload: dict) -> None:
    """
    Child process: pin to one physical GPU via CUDA_VISIBLE_DEVICES, load full 3B on cuda:0,
    then run all assigned (layer, λ) jobs sequentially.
    """
    import os

    gpu_id = int(payload["gpu_id"])
    tqdm_position = int(payload.get("tqdm_position", gpu_id % 8))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import importlib.util as iu
    import json as json_mod
    import math
    import sys as sys_mod
    import torch as th
    from torch.nn import CrossEntropyLoss as CEL
    from tqdm import tqdm as tqdm_mod
    from transformers import AutoModelForCausalLM as AM, AutoTokenizer as AT

    base = Path(payload["base"])
    cache_dir = Path(payload["cache_dir"])
    cfg_model = payload["model"]
    max_length = int(payload["max_length"])
    max_docs = payload.get("max_docs")
    no_cache = bool(payload["no_cache"])
    dtype_str = payload["dtype"]
    prm_only = bool(payload.get("prm_only", False))
    ppl_mode = bool(payload.get("ppl_mode", False))

    cgn_path = base / "new_scripts/my_scripts/control_generation_nll.py"
    spec = iu.spec_from_file_location("control_generation_nll", cgn_path)
    mod = iu.module_from_spec(spec)
    sys_mod.modules["control_generation_nll"] = mod
    spec.loader.exec_module(mod)
    load_records_dedup = mod.load_records_dedup
    extract_prompt_from_record = mod.extract_prompt_from_record
    extract_generation = mod.extract_generation
    nll_one = None if (prm_only or ppl_mode) else mod.nll_one

    dt = th.float16 if dtype_str == "float16" else th.bfloat16
    tokenizer = AT.from_pretrained(cfg_model, trust_remote_code=True)
    model = AM.from_pretrained(
        cfg_model,
        torch_dtype=dt,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
    )
    model.eval()

    def nll_prm_local(
        prompt: str,
        continuation: str,
    ) -> tuple[float | None, int]:
        if not continuation.strip():
            return None, 0
        pin = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        rin = tokenizer(
            continuation,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        prompt_ids = pin.input_ids.to(model.device)
        response_ids = rin.input_ids.to(model.device)
        input_ids = th.cat([prompt_ids, response_ids], dim=1)
        if input_ids.shape[1] > max_length:
            return None, 0
        start_idx = int(prompt_ids.shape[1])
        if start_idx == 0 or input_ids.shape[1] <= start_idx:
            return None, 0
        lf = CEL(reduction="mean")
        with th.inference_mode():
            out = model(input_ids)
            logits = out.logits
        shift_logits = logits[:, start_idx - 1 : -1, :]
        shift_labels = input_ids[:, start_idx:]
        if shift_logits.shape[1] != shift_labels.shape[1]:
            return None, int(shift_labels.numel())
        loss = lf(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        if th.isnan(loss) or th.isinf(loss):
            return None, int(shift_labels.numel())
        return float(loss.item()), int(shift_labels.numel())

    cache_dir.mkdir(parents=True, exist_ok=True)

    for jd in payload["jobs"]:
        layer = int(jd["layer"])
        lam = float(jd["lambda"])
        run_dir = Path(jd["run_dir"])
        samples_path = Path(jd["samples_path"])

        lam_s = f"{lam:g}".replace("-", "neg").replace(".", "p")
        cp = cache_dir / f"L{layer}_lam{lam_s}.json"
        if cp.is_file() and not no_cache:
            print(f"[GPU{gpu_id}] cache hit {cp.name}", flush=True)
            continue

        records = load_records_dedup(samples_path)
        if max_docs is not None:
            records = records[: int(max_docs)]

        ctrl_nlls: list[float] = []
        ctrl_toks: list[int] = []
        prm_nlls: list[float] = []
        prm_toks: list[int] = []
        ppl_vals: list[float] = []
        n_skip_ctrl = 0
        n_skip_prm = 0
        n_skip_ppl = 0

        for rec in tqdm_mod(
            records,
            desc=f"GPU{gpu_id} L{layer} λ={lam:g}",
            position=tqdm_position,
            leave=True,
        ):
            prompt = extract_prompt_from_record(rec)
            gen = extract_generation(rec)
            if prompt is None:
                n_skip_ctrl += 1
                n_skip_prm += 1
                n_skip_ppl += 1
                continue

            if ppl_mode:
                # PPL on response-only text: compute_ppl_and_rank(generated_text)
                # Reference prm_results uses generated_text = response only (no prompt).
                text_for_ppl = gen if gen and gen.strip() else None
                if not text_for_ppl:
                    n_skip_ppl += 1
                    continue
                inputs = tokenizer(text_for_ppl, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
                with th.inference_mode():
                    outputs = model(**inputs)
                    logits = outputs.logits[:, :-1, :]
                    labels = inputs.input_ids[:, 1:]
                    shift_logits = logits.reshape(-1, logits.size(-1))
                    shift_labels = labels.reshape(-1)
                    loss_fct_local = CEL(reduction="none")
                    loss = loss_fct_local(shift_logits, shift_labels)
                    ppl_val = th.exp(loss.mean()).item()
                if math.isnan(ppl_val) or math.isinf(ppl_val):
                    n_skip_ppl += 1
                else:
                    ppl_vals.append(ppl_val)
            else:
                if not prm_only:
                    nm, nt = nll_one(prompt, gen, model, tokenizer, max_length)
                    if nm is None:
                        n_skip_ctrl += 1
                    else:
                        ctrl_nlls.append(nm)
                        ctrl_toks.append(nt)

                pm, pt = nll_prm_local(prompt, gen)
                if pm is None:
                    n_skip_prm += 1
                else:
                    prm_nlls.append(pm)
                    prm_toks.append(pt)

        ctrl_wm = _weighted_mean(ctrl_nlls, ctrl_toks) if (not prm_only and not ppl_mode) else None
        prm_wm = _weighted_mean(prm_nlls, prm_toks) if not ppl_mode else None
        ppl_mean = (sum(ppl_vals) / len(ppl_vals)) if ppl_vals else None

        blob = {
            "layer": layer,
            "lambda": lam,
            "run_dir": str(run_dir.relative_to(base)),
            "samples_jsonl": str(samples_path.relative_to(base)),
            "n_docs": len(records),
            "control_labels_mask": ctrl_wm,
            "prm_prompt_resp_concat": prm_wm,
            "ppl_full_text": ppl_mean,
            "control_mean_of_seq_means": (sum(ctrl_nlls) / len(ctrl_nlls)) if ctrl_nlls else None,
            "prm_mean_of_seq_means": (sum(prm_nlls) / len(prm_nlls)) if prm_nlls else None,
            "n_scored_control": len(ctrl_nlls),
            "n_scored_prm": len(prm_nlls),
            "n_scored_ppl": len(ppl_vals),
            "n_skip_control": n_skip_ctrl,
            "n_skip_prm": n_skip_prm,
            "n_skip_ppl": n_skip_ppl,
            "max_length": max_length,
            "model": cfg_model,
            "gpu_id": gpu_id,
            "prm_only_model_evaluator_style": prm_only,
            "ppl_mode": ppl_mode,
        }
        cp.write_text(json_mod.dumps(blob, indent=2), encoding="utf-8")
        summary_keys = ["gpu", "layer", "lambda"]
        summary = {"gpu": gpu_id, "layer": layer, "lambda": lam}
        if ppl_mode:
            summary["ppl_full_text"] = ppl_mean
        else:
            summary["control_labels_mask"] = ctrl_wm
            summary["prm_prompt_resp_concat"] = prm_wm
        print(json_mod.dumps(summary), flush=True)


def write_nll_plots(
    series: dict[str, dict[int, dict[float, float]]],
    lam_sorted: list[float],
    target_layers: tuple[int, ...],
    out_dir: Path,
    plot_stem: str,
    prm_only: bool,
    ppl_mode: bool = False,
) -> None:
    from matplotlib.ticker import MultipleLocator

    colors = {35: "#1a237e", 31: "#3949ab", 17: "#64b5f6"}
    labels = {35: "Qwen2.5-3B-Instruct-L35", 31: "Qwen2.5-3B-Instruct-L31", 17: "Qwen2.5-3B-Instruct-L17"}
    layer_draw_order = tuple(L for L in (35, 31, 17) if L in target_layers)
    tick_step = 2.5 if max(lam_sorted) - min(lam_sorted) > 15 else 2.0

    def _style_axis(ax) -> None:
        ax.set_xlabel(r"$\lambda$")
        ax.set_xlim(min(lam_sorted) - 0.5, max(lam_sorted) + 0.5)
        ax.xaxis.set_major_locator(MultipleLocator(tick_step))
        ax.grid(True, alpha=0.35)

    def plot_method(method_key: str, title_suffix: str, out_name: str, caption: str, ylabel: str = "Negative Log-Likelihood (ln)") -> None:
        plt.figure(figsize=(8.5, 5.4))
        for L in layer_draw_order:
            ys = [series[method_key][L].get(l, float("nan")) for l in lam_sorted]
            plt.plot(
                lam_sorted,
                ys,
                marker="o",
                linewidth=1.5,
                markersize=3 if len(lam_sorted) > 20 else 5,
                color=colors[L],
                label=labels[L],
            )
        plt.ylabel(ylabel)
        _style_axis(plt.gca())
        plt.legend(loc="upper right", fontsize=8)
        plt.figtext(0.5, 0.02, caption, ha="center", fontsize=7)
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        outp = out_dir / out_name
        plt.savefig(outp, dpi=200)
        plt.close()
        print(f"wrote {outp}")

    if ppl_mode:
        plot_method(
            "ppl_full_text",
            "compute_ppl_and_rank (full-text PPL)",
            f"{plot_stem}_ppl_full_text.png",
            "Full-text PPL = exp(mean CE over all tokens). Same as ModelEvaluator.compute_ppl_and_rank in 06_PRM_plots_2.py. "
            "eval_grid_less_tokens_3B_lib_manual_same_lib_1000samples/gsm8k_cot_zeroshot.",
            ylabel="Perplexity (PPL)",
        )
        return

    if prm_only:
        plot_method(
            "prm_prompt_resp_concat",
            "06_PRM_plots_2 ModelEvaluator (prompt True + resp False)",
            f"{plot_stem}_prm_model_evaluator.png",
            "Mean CE on continuation tokens only; same tokenization as ModelEvaluator.compute_rank_shift in 06_PRM_plots_2.py. "
            "Corpus-weighted over generated tokens. eval_grid_less_tokens_3B_lib_manual_same_lib_1000samples/gsm8k_cot_zeroshot.",
        )
        return

    plot_method(
        "control_labels_mask",
        "control_generation_nll",
        f"{plot_stem}_control_labels_mask.png",
        "Average token-level NLL (cross-entropy, nats) under the base model vs λ; "
        "same token alignment as new_scripts/my_scripts/control_generation_nll.py (masked prompt). "
        "Corpus-weighted over generated tokens.",
    )
    plot_method(
        "prm_prompt_resp_concat",
        "03/06 PRM concat",
        f"{plot_stem}_prm_concat.png",
        "Average token-level NLL (cross-entropy, nats) under the base model vs λ; "
        "same token alignment as ModelEvaluator.compute_rank_shift. Corpus-weighted over generated tokens.",
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    for ax, (method_key, suffix) in zip(
        axes,
        [
            ("control_labels_mask", "control_generation_nll"),
            ("prm_prompt_resp_concat", "03/06 PRM concat"),
        ],
    ):
        for L in layer_draw_order:
            ys = [series[method_key][L].get(l, float("nan")) for l in lam_sorted]
            ax.plot(
                lam_sorted,
                ys,
                marker="o",
                linewidth=1.5,
                markersize=3 if len(lam_sorted) > 20 else 5,
                color=colors[L],
                label=labels[L],
            )
        _style_axis(ax)
    axes[0].set_ylabel("Negative Log-Likelihood (ln)")
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.text(
        0.5,
        0.02,
        "Steered GSM8K CoT (1000 samples / config), Qwen2.5-3B-Instruct base; left = control_generation_nll, right = PRM concat.",
        ha="center",
        fontsize=7,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    combo = out_dir / f"{plot_stem}_both_methods.png"
    fig.savefig(combo, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"wrote {combo}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--no-cache", action="store_true", help="Ignore / overwrite per-job JSON cache.")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only read cache JSONs and write plots + grid_eval (no GPU). Use same --cache-root / --run-tag as the compute run.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated physical GPU indices. Each GPU runs one process with a full model copy.",
    )
    parser.add_argument("--lambda-min", type=float, default=-10.0)
    parser.add_argument("--lambda-max", type=float, default=10.0)
    parser.add_argument(
        "--lambda-step",
        type=float,
        default=2.5,
        help="Grid step for λ (e.g. 0.5 with [-10,10] → 41 points). Default 2.5 preserves original coarse grid.",
    )
    parser.add_argument(
        "--prm-only",
        action="store_true",
        help="Only compute 06_PRM_plots_2 ModelEvaluator-style NLL (prompt add_special_tokens=True, response False); skip control_generation_nll.",
    )
    parser.add_argument(
        "--ppl-mode",
        action="store_true",
        help="Compute full-text PPL (compute_ppl_and_rank style): tokenizer(prompt+gen) → CE over ALL tokens → exp(mean). "
             "This matches the 'ppl' field in prm_results/results_merged.json and the original Figure 6 y-axis.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Directory for per-(layer,λ) JSON caches. Default: documents/figure6_nll_cache or documents/<run-tag>/cache",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="If set, cache goes to documents/<run-tag>/cache/ and outputs use stem <run-tag> unless --plot-stem is set.",
    )
    parser.add_argument(
        "--plot-stem",
        type=str,
        default=None,
        help="Stem for *_grid_eval.json / *_meta.json / plot PNGs (default: run-tag, or figure6_nll).",
    )
    args = parser.parse_args()

    DOCS.mkdir(parents=True, exist_ok=True)
    target_lambdas = build_lambda_grid(args.lambda_min, args.lambda_max, args.lambda_step)
    target_layers = DEFAULT_TARGET_LAYERS

    if args.run_tag:
        cache_root = DOCS / args.run_tag / "cache"
        plot_stem = args.plot_stem or args.run_tag
    elif args.cache_root is not None:
        cache_root = Path(args.cache_root)
        plot_stem = args.plot_stem or "nll_grid"
    else:
        cache_root = DOCS / "figure6_nll_cache"
        plot_stem = args.plot_stem or "figure6_nll"

    cache_root.mkdir(parents=True, exist_ok=True)
    out_json = DOCS / f"{plot_stem}_grid_eval.json"
    meta_path = DOCS / f"{plot_stem}_meta.json"

    if args.plot_only:
        if meta_path.is_file():
            file_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            target_lambdas = file_meta["target_lambdas"]
            target_layers = tuple(file_meta["layers"])
            prm_only_plot = file_meta.get("prm_only", args.prm_only)
            ppl_mode_plot = file_meta.get("ppl_mode", args.ppl_mode)
        else:
            prm_only_plot = args.prm_only
            ppl_mode_plot = args.ppl_mode
        series = load_series_from_cache(cache_root, target_lambdas, target_layers, prm_only_plot, ppl_mode_plot)
        meta_out = {
            "lambda_min": target_lambdas[0] if target_lambdas else args.lambda_min,
            "lambda_max": target_lambdas[-1] if target_lambdas else args.lambda_max,
            "lambda_step": args.lambda_step,
            "target_lambdas": target_lambdas,
            "layers": list(target_layers),
            "prm_only": prm_only_plot,
            "ppl_mode": ppl_mode_plot,
            "cache_root": str(cache_root.relative_to(BASE)),
        }
        payload = {"_meta": meta_out, **series}
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if not args.skip_plots:
            write_nll_plots(series, target_lambdas, target_layers, DOCS, plot_stem, prm_only_plot, ppl_mode_plot)
        return

    cgn = _load_control_generation_nll()
    load_records_dedup = cgn.load_records_dedup
    extract_prompt_from_record = cgn.extract_prompt_from_record
    extract_generation = cgn.extract_generation
    nll_one = cgn.nll_one

    jobs = discover_jobs(target_lambdas, target_layers)
    expect_n = len(target_layers) * len(target_lambdas)
    if len(jobs) != expect_n:
        print(f"Warning: expected {expect_n} jobs, found {len(jobs)}", file=sys.stderr)

    gpu_list = [int(x.strip()) for x in args.gpus.split(",") if x.strip() != ""]
    if not gpu_list:
        print("Empty --gpus", file=sys.stderr)
        sys.exit(1)

    if args.ppl_mode:
        keys = ["ppl_full_text"]
    elif args.prm_only:
        keys = ["prm_prompt_resp_concat"]
    else:
        keys = ["control_labels_mask", "prm_prompt_resp_concat"]
    series: dict[str, dict[int, dict[float, float]]] = {k: {L: {} for L in target_layers} for k in keys}

    pending: list[dict] = []
    for layer, lam, run_dir, samples_path in jobs:
        cp = cache_path(layer, lam, cache_root)
        if cp.is_file() and not args.no_cache:
            blob = json.loads(cp.read_text(encoding="utf-8"))
            for key in keys:
                v = blob.get(key)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    series[key][layer][lam] = float(v)
            print(f"cache hit {cp.name}")
            continue
        pending.append(_job_to_dict(layer, lam, run_dir, samples_path))

    if pending:
        if len(gpu_list) > 1:
            buckets: list[list[dict]] = [[] for _ in gpu_list]
            for i, jd in enumerate(pending):
                buckets[i % len(gpu_list)].append(jd)
            ctx = mp.get_context("spawn")
            procs: list[mp.Process] = []
            for wi, gid in enumerate(gpu_list):
                sub = buckets[wi]
                if not sub:
                    continue
                payload = {
                    "gpu_id": gid,
                    "tqdm_position": wi,
                    "jobs": sub,
                    "base": str(BASE),
                    "cache_dir": str(cache_root),
                    "model": args.model,
                    "max_length": args.max_length,
                    "max_docs": args.max_docs,
                    "no_cache": args.no_cache,
                    "dtype": args.dtype,
                    "prm_only": args.prm_only,
                    "ppl_mode": args.ppl_mode,
                }
                p = ctx.Process(target=_gpu_worker_run, args=(payload,))
                p.start()
                procs.append(p)
            for p in procs:
                p.join()
                if p.exitcode != 0:
                    print(f"Worker exited with code {p.exitcode}", file=sys.stderr)
                    sys.exit(p.exitcode if p.exitcode is not None else 1)
        else:
            dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=dtype,
                device_map={"": "cuda:0"},
                trust_remote_code=True,
            )
            model.eval()

            for jd in pending:
                layer = jd["layer"]
                lam = jd["lambda"]
                run_dir = Path(jd["run_dir"])
                samples_path = Path(jd["samples_path"])
                cp = cache_path(layer, lam, cache_root)

                records = load_records_dedup(samples_path)
                if args.max_docs is not None:
                    records = records[: args.max_docs]

                ctrl_nlls: list[float] = []
                ctrl_toks: list[int] = []
                prm_nlls: list[float] = []
                prm_toks: list[int] = []
                n_skip_ctrl = 0
                n_skip_prm = 0

                for rec in tqdm(records, desc=f"L{layer} λ={lam:g}"):
                    prompt = extract_prompt_from_record(rec)
                    gen = extract_generation(rec)
                    if prompt is None:
                        n_skip_ctrl += 1
                        n_skip_prm += 1
                        continue

                    if not args.prm_only:
                        nm, nt = nll_one(prompt, gen, model, tokenizer, args.max_length)
                        if nm is None:
                            n_skip_ctrl += 1
                        else:
                            ctrl_nlls.append(nm)
                            ctrl_toks.append(nt)

                    pm, pt = nll_prm_concat_style(prompt, gen, model, tokenizer, args.max_length)
                    if pm is None:
                        n_skip_prm += 1
                    else:
                        prm_nlls.append(pm)
                        prm_toks.append(pt)

                ctrl_wm = _weighted_mean(ctrl_nlls, ctrl_toks) if not args.prm_only else None
                prm_wm = _weighted_mean(prm_nlls, prm_toks)

                blob = {
                    "layer": layer,
                    "lambda": lam,
                    "run_dir": str(run_dir.relative_to(BASE)),
                    "samples_jsonl": str(samples_path.relative_to(BASE)),
                    "n_docs": len(records),
                    "control_labels_mask": ctrl_wm,
                    "prm_prompt_resp_concat": prm_wm,
                    "control_mean_of_seq_means": (sum(ctrl_nlls) / len(ctrl_nlls)) if ctrl_nlls else None,
                    "prm_mean_of_seq_means": (sum(prm_nlls) / len(prm_nlls)) if prm_nlls else None,
                    "n_scored_control": len(ctrl_nlls),
                    "n_scored_prm": len(prm_nlls),
                    "n_skip_control": n_skip_ctrl,
                    "n_skip_prm": n_skip_prm,
                    "max_length": args.max_length,
                    "model": args.model,
                    "prm_only_model_evaluator_style": args.prm_only,
                }
                cp.write_text(json.dumps(blob, indent=2), encoding="utf-8")
                print(
                    json.dumps(
                        {k: blob[k] for k in ("layer", "lambda", "control_labels_mask", "prm_prompt_resp_concat")}
                    )
                )

                if ctrl_wm is not None and "control_labels_mask" in series:
                    series["control_labels_mask"][layer][lam] = ctrl_wm
                if prm_wm is not None:
                    series["prm_prompt_resp_concat"][layer][lam] = prm_wm

    for layer in target_layers:
        for lam in target_lambdas:
            cp = cache_path(layer, lam, cache_root)
            if not cp.is_file():
                continue
            blob = json.loads(cp.read_text(encoding="utf-8"))
            for key in keys:
                v = blob.get(key)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    series[key][layer][lam] = float(v)

    meta = {
        "lambda_min": args.lambda_min,
        "lambda_max": args.lambda_max,
        "lambda_step": args.lambda_step,
        "target_lambdas": target_lambdas,
        "layers": list(target_layers),
        "prm_only": args.prm_only,
        "ppl_mode": args.ppl_mode,
        "cache_root": str(cache_root.relative_to(BASE)),
        "grid": str(GRID.relative_to(BASE)),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    out_payload = {"_meta": meta, **series}
    out_json.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    if args.skip_plots:
        return

    write_nll_plots(series, target_lambdas, target_layers, DOCS, plot_stem, args.prm_only, args.ppl_mode)


if __name__ == "__main__":
    # Avoid re-running main() in multiprocessing spawn children (they re-import this file as __main__).
    if mp.parent_process() is None:
        main()
