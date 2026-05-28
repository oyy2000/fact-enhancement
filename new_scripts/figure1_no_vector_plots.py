#!/usr/bin/env python3
"""
Figure 1 (tokens/step / step count vs accuracy) from exps/no_vector GSM8K samples.

Uses the same step splitting and per-step token counts as 07_run_prm_single.py
(but does not run the PRM model — only step_token_len + exact_match are needed
for plots_concise.plot_avg_tokens_vs_acc / plot_avg_steps_vs_acc).

Default: x-axis metrics averaged over correct samples only (Y==1); y-axis is
overall accuracy on the PRM-eval subset (non-strict-match lines), matching
07_run_prm_single filtering.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
NO_VECTOR_GSM8K = ROOT / "exps" / "no_vector" / "gsm8k_cot_zeroshot_unified"
DEFAULT_OUT = NO_VECTOR_GSM8K / "figure1_plots"


def split_steps_for_qwen(cot_text: str, strategy: str = "auto") -> list[str]:
    """Mirrors 07_run_prm_single.split_steps_for_qwen."""
    if not cot_text:
        return []
    cot_text = cot_text.strip()
    separator = None
    if strategy == "double_newline":
        separator = "\n\n"
    elif strategy == "single_newline":
        separator = "\n"
    elif strategy == "auto":
        if "\n\n" in cot_text:
            separator = "\n\n"
        elif "\n" in cot_text:
            separator = "\n"
    if separator:
        parts = cot_text.split(separator)
        steps = []
        for i, p in enumerate(parts):
            if not p.strip():
                continue
            if i < len(parts) - 1:
                steps.append(p.strip() + separator)
            else:
                steps.append(p.strip())
        return steps
    steps = re.split(r"(?<=[.!?])\s+", cot_text)
    return [s.strip() for s in steps if s.strip()]


def load_plots_concise():
    spec = importlib.util.spec_from_file_location(
        "plots_concise",
        str(ROOT / "05_plots_concise.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["plots_concise"] = mod
    spec.loader.exec_module(mod)
    return mod


def discover_runs(task_dir: Path) -> list[tuple[str, str, Path]]:
    """
    Returns list of (short_model_name, hf_model_id, jsonl_path).
    """
    runs: list[tuple[str, str, Path]] = []
    if not task_dir.is_dir():
        return runs
    for child in sorted(task_dir.iterdir()):
        if not child.is_dir() or not child.name.endswith("_no_vector"):
            continue
        subs = [d for d in child.iterdir() if d.is_dir() and "__" in d.name]
        if not subs:
            continue
        model_sub = subs[0]
        hf_id = model_sub.name.replace("__", "/")
        short = hf_id.split("/")[-1]
        jsonls = sorted(model_sub.glob("samples_*.jsonl"))
        if not jsonls:
            continue
        runs.append((short, hf_id, jsonls[-1]))
    return runs


def build_entry_from_jsonl(
    jsonl_path: Path,
    hf_model_id: str,
    *,
    label_key: str = "exact_match",
) -> dict | None:
    tok = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True, use_fast=True)
    Y: list[int] = []
    step_token_len: list[list[int]] = []
    step_scores: list[list[float]] = []

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("filter") == "strict-match":
                continue
            cot = (d.get("resps") or [[""]])[0][0]
            if not isinstance(cot, str):
                continue
            cot = cot.strip()
            steps = split_steps_for_qwen(cot)
            if not steps:
                continue
            lens = [len(tok.encode(s, add_special_tokens=False)) for s in steps]
            step_token_len.append(lens)
            step_scores.append([0.5] * len(lens))
            yv = d.get(label_key, 0)
            Y.append(int(yv))

    if not Y:
        return None
    return {
        "file_used": str(jsonl_path),
        "gen_model": hf_model_id,
        "Y": Y,
        "step_scores": step_scores,
        "step_token_len": step_token_len,
    }


def build_model_results(runs: list[tuple[str, str, Path]], label_key: str) -> dict:
    out: dict = {}
    for short, hf_id, jp in runs:
        entry = build_entry_from_jsonl(jp, hf_id, label_key=label_key)
        if entry is None:
            print(f"[skip] no samples: {short} ({jp})")
            continue
        out[short] = {"L0": {"BASELINE": entry}}
        n = len(entry["Y"])
        acc = float(np.mean(entry["Y"]))
        print(f"  {short}: n={n} acc={acc:.4f} ({jp.name})")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task_dir",
        type=Path,
        default=NO_VECTOR_GSM8K,
        help="e.g. exps/no_vector/gsm8k_cot_zeroshot_unified",
    )
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--all_samples_x",
        action="store_true",
        help="Average x-axis over all (non-strict) samples instead of correct-only.",
    )
    p.add_argument("--label_key", type=str, default="exact_match")
    args = p.parse_args()

    runs = discover_runs(args.task_dir)
    if not runs:
        print(f"No runs under {args.task_dir}")
        sys.exit(1)

    print(f"Building model_results from {len(runs)} runs under {args.task_dir}...")
    model_results = build_model_results(runs, args.label_key)
    if not model_results:
        print("No model results built.")
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("correct_wrong", "scatter", "special", "all", "per_step_avg"):
        (args.out_dir / sub).mkdir(parents=True, exist_ok=True)

    plots_concise = load_plots_concise()
    plots_concise.setup_plotting(model_results, str(args.out_dir))
    correct_only = not args.all_samples_x
    plots_concise.plot_avg_tokens_vs_acc(correct_only=correct_only)
    plots_concise.plot_avg_steps_vs_acc(correct_only=correct_only)
    print("Done. Figures under:", args.out_dir / "special")


if __name__ == "__main__":
    main()
