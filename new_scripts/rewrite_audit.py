#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rewrite Audit: automatic metrics + GPT-5.1 dual-judge evaluation
for (resp_before, resp_after) pairs in a rewrite JSON file.

Usage:
    python new_scripts/rewrite_audit.py \
        --input_json exps/gpt_rewrites_unified_new/Qwen_Qwen2.5-3B-Instruct/rewritten_old.json \
        --workers 8

    # Dry-run (auto metrics only, no API calls):
    python new_scripts/rewrite_audit.py --dry_run ...
"""

import argparse
import json
import os
import re
import sys
import time
import random
import difflib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from utils import split_auto, split_double_newline, split_single_newline_robust

DEFAULT_INPUT = str(
    BASE / "exps" / "gpt_rewrites_unified_new"
    / "Qwen_Qwen2.5-3B-Instruct" / "rewritten_old.json"
)

# ---------------------------------------------------------------------------
# Step / token helpers
# ---------------------------------------------------------------------------

def count_steps(text: str) -> int:
    return len(split_auto(text)) if text.strip() else 0


def tokenize(text: str, tokenizer) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def density(text: str, tokenizer) -> float:
    """tokens-per-step (ρ)."""
    steps = split_auto(text)
    if not steps:
        return 0.0
    total_tok = len(tokenize(text, tokenizer))
    return total_tok / len(steps)

# ---------------------------------------------------------------------------
# Automatic metrics
# ---------------------------------------------------------------------------

_NUM_OP_RE = re.compile(r"[\d]+(?:\.\d+)?|[+\-*/=]")


def extract_nums_ops(text: str) -> List[str]:
    return _NUM_OP_RE.findall(text)


def jaccard_token_overlap(a: str, b: str) -> float:
    wa = set(a.split())
    wb = set(b.split())
    if not wa and not wb:
        return 1.0
    inter = wa & wb
    union = wa | wb
    return len(inter) / len(union)


def normalised_edit_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def adjacent_merge_ratio(before: str, after: str) -> float:
    """Fraction of *before* steps consumed by adjacent merges."""
    steps_b = split_auto(before)
    steps_a = split_auto(after)
    if not steps_b:
        return 0.0

    # Use SequenceMatcher on step-level to find merge patterns
    sm = difflib.SequenceMatcher(
        None,
        [s.strip() for s in steps_b],
        [s.strip() for s in steps_a],
    )
    merged_before_steps = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            before_span = i2 - i1
            after_span = j2 - j1
            # Adjacent merge: multiple before-steps -> fewer after-steps
            if before_span > after_span:
                merged_before_steps += before_span
        elif tag == "delete":
            merged_before_steps += (i2 - i1)
    return merged_before_steps / len(steps_b)


def changed_nums_ops_ratio(before: str, after: str) -> float:
    """Fraction of numbers/operators that changed between before and after."""
    nb = extract_nums_ops(before)
    na = extract_nums_ops(after)
    if not nb and not na:
        return 0.0
    cb = Counter(nb)
    ca = Counter(na)
    all_keys = set(cb.keys()) | set(ca.keys())
    total = sum(max(cb[k], ca[k]) for k in all_keys)
    diff = sum(abs(cb[k] - ca[k]) for k in all_keys)
    return diff / total if total else 0.0


def compute_auto_metrics(entry: Dict, tokenizer) -> Dict[str, Any]:
    before = entry.get("resp_before", "")
    after = entry.get("resp_after", "")

    steps_b = count_steps(before)
    steps_a = count_steps(after)
    dens_b = density(before, tokenizer)
    dens_a = density(after, tokenizer)

    return {
        "doc_id": entry.get("doc_id"),
        "step_count_before": steps_b,
        "step_count_after": steps_a,
        "step_count_delta": steps_a - steps_b,
        "density_before": round(dens_b, 2),
        "density_after": round(dens_a, 2),
        "density_delta": round(dens_a - dens_b, 2),
        "token_overlap_jaccard": round(jaccard_token_overlap(before, after), 4),
        "edit_similarity": round(normalised_edit_similarity(before, after), 4),
        "adjacent_merge_ratio": round(adjacent_merge_ratio(before, after), 4),
        "changed_nums_ops_ratio": round(changed_nums_ops_ratio(before, after), 4),
    }

# ---------------------------------------------------------------------------
# GPT-5.1 dual judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = "You are an expert math-reasoning auditor. Respond ONLY with valid JSON."

JUDGE_PROMPT_TEMPLATE = """You are comparing an ORIGINAL math solution with a REWRITTEN version.

**Question:**
{question}

**Original solution (before):**
{before}

**Rewritten solution (after):**
{after}

Judge the rewrite on these 6 dimensions. Answer each as true or false.

1. final_answer_preserved: Is the final boxed/numeric answer identical in both?
2. reasoning_meaning_preserved: Is the intermediate reasoning semantically equivalent (same logical steps, same conclusions)?
3. new_facts_introduced: Does the rewrite introduce any facts, calculations, or claims NOT present in the original?
4. error_fixed: Does the rewrite fix a mathematical error that existed in the original?
5. mainly_adjacent_merge: Are the changes primarily merging adjacent steps (rather than large-scale rewriting)?
6. style_preserved: Is the formatting and writing style basically the same?

Respond with ONLY a JSON object (no markdown fences):
{{"final_answer_preserved": bool, "reasoning_meaning_preserved": bool, "new_facts_introduced": bool, "error_fixed": bool, "mainly_adjacent_merge": bool, "style_preserved": bool}}"""


def call_judge(
    client: OpenAI,
    model: str,
    question: str,
    before: str,
    after: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    retries: int = 4,
    sleep_base: float = 1.5,
) -> Dict[str, bool]:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, before=before, after=after
    )
    last_err = None
    for attempt in range(retries):
        try:
            try:
                resp = client.responses.create(
                    model=model,
                    instructions=JUDGE_SYSTEM,
                    input=prompt,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                raw = (resp.output_text or "").strip()
            except AttributeError:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                raw = resp.choices[0].message.content.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)

        except Exception as e:
            last_err = e
            time.sleep(sleep_base * (2 ** attempt))

    raise RuntimeError(f"Judge API failed after {retries} retries: {last_err}")


JUDGE_KEYS = [
    "final_answer_preserved",
    "reasoning_meaning_preserved",
    "new_facts_introduced",
    "error_fixed",
    "mainly_adjacent_merge",
    "style_preserved",
]


def dual_judge_one(
    client: OpenAI, model: str, entry: Dict
) -> Dict[str, Any]:
    question = (entry.get("doc") or {}).get("question", "")
    before = entry.get("resp_before", "")
    after = entry.get("resp_after", "")

    j1 = call_judge(client, model, question, before, after)
    j2 = call_judge(client, model, question, before, after)

    majority = {}
    agreement = {}
    for k in JUDGE_KEYS:
        v1 = bool(j1.get(k))
        v2 = bool(j2.get(k))
        agreement[k] = v1 == v2
        majority[k] = v1 if v1 == v2 else v1  # on disagree, take judge-1

    return {
        "judge_1": {k: j1.get(k) for k in JUDGE_KEYS},
        "judge_2": {k: j2.get(k) for k in JUDGE_KEYS},
        "majority": majority,
        "agreement": agreement,
    }

# ---------------------------------------------------------------------------
# Aggregation & report
# ---------------------------------------------------------------------------

def aggregate_report(results: List[Dict]) -> Dict:
    n = len(results)

    # --- auto metrics ---
    auto_keys = [
        "step_count_delta", "density_delta",
        "token_overlap_jaccard", "edit_similarity",
        "adjacent_merge_ratio", "changed_nums_ops_ratio",
    ]
    auto_summary = {}
    for k in auto_keys:
        vals = [r["auto"][k] for r in results]
        auto_summary[k] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
        }

    # extra: step count before/after means
    auto_summary["step_count_before_mean"] = round(
        float(np.mean([r["auto"]["step_count_before"] for r in results])), 2
    )
    auto_summary["step_count_after_mean"] = round(
        float(np.mean([r["auto"]["step_count_after"] for r in results])), 2
    )
    auto_summary["density_before_mean"] = round(
        float(np.mean([r["auto"]["density_before"] for r in results])), 2
    )
    auto_summary["density_after_mean"] = round(
        float(np.mean([r["auto"]["density_after"] for r in results])), 2
    )

    # --- judge metrics ---
    judge_summary = {}
    has_judge = "judge" in results[0]
    if has_judge:
        for k in JUDGE_KEYS:
            votes = [r["judge"]["majority"][k] for r in results]
            agrees = [r["judge"]["agreement"][k] for r in results]
            judge_summary[k] = {
                "true_pct": round(sum(1 for v in votes if v) / n * 100, 1),
                "agreement_pct": round(sum(1 for a in agrees if a) / n * 100, 1),
            }

    return {
        "n_samples": n,
        "auto_metrics": auto_summary,
        "judge_metrics": judge_summary,
    }


def print_report(report: Dict):
    n = report["n_samples"]
    print(f"\n{'='*70}")
    print(f"  REWRITE AUDIT REPORT  (n={n})")
    print(f"{'='*70}")

    print(f"\n--- Automatic Metrics ---")
    print(f"  Steps  : {report['auto_metrics']['step_count_before_mean']:.1f} -> "
          f"{report['auto_metrics']['step_count_after_mean']:.1f}  "
          f"(delta mean={report['auto_metrics']['step_count_delta']['mean']:+.2f})")
    print(f"  Density: {report['auto_metrics']['density_before_mean']:.1f} -> "
          f"{report['auto_metrics']['density_after_mean']:.1f}  "
          f"(delta mean={report['auto_metrics']['density_delta']['mean']:+.2f})")

    fmt = "  {:<28s}  mean={:>7.4f}  std={:>7.4f}  min={:>7.4f}  max={:>7.4f}"
    for k in ["token_overlap_jaccard", "edit_similarity",
              "adjacent_merge_ratio", "changed_nums_ops_ratio"]:
        s = report["auto_metrics"][k]
        print(fmt.format(k, s["mean"], s["std"], s["min"], s["max"]))

    if report["judge_metrics"]:
        print(f"\n--- GPT Judge (dual, majority vote) ---")
        print(f"  {'Dimension':<32s}  {'True%':>6s}  {'Agree%':>7s}")
        print(f"  {'-'*32}  {'-'*6}  {'-'*7}")
        for k in JUDGE_KEYS:
            s = report["judge_metrics"][k]
            print(f"  {k:<32s}  {s['true_pct']:>5.1f}%  {s['agreement_pct']:>6.1f}%")

    print(f"\n{'='*70}\n")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Rewrite audit: auto metrics + GPT dual judge")
    ap.add_argument("--input_json", default=DEFAULT_INPUT)
    ap.add_argument("--output_json", default=None,
                    help="Output path (default: sibling audit_results.json)")
    ap.add_argument("--model", default="gpt-5.1", help="Judge model")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample_n", type=int, default=200,
                    help="Number of samples to audit (default 200)")
    ap.add_argument("--last_n", type=int, default=0,
                    help="If >0, take the last N samples instead of random sampling")
    ap.add_argument("--dry_run", action="store_true",
                    help="Compute auto metrics only, skip GPT judge")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B-Instruct")
    args = ap.parse_args()

    # --- load data ---
    print(f"Loading {args.input_json} ...", file=sys.stderr)
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} entries.", file=sys.stderr)

    # --- sample ---
    random.seed(args.seed)
    if args.last_n > 0:
        data = data[-args.last_n:]
        print(f"  Took last {len(data)} entries.", file=sys.stderr)
    elif len(data) > args.sample_n:
        data = random.sample(data, args.sample_n)
        print(f"  Sampled {args.sample_n} entries (seed={args.seed}).", file=sys.stderr)

    # --- tokenizer ---
    print(f"Loading tokenizer {args.tokenizer} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # --- auto metrics ---
    print("Computing automatic metrics ...", file=sys.stderr)
    results = []
    for entry in tqdm(data, desc="Auto metrics"):
        auto = compute_auto_metrics(entry, tokenizer)
        results.append({"doc_id": entry.get("doc_id"), "auto": auto})

    # --- GPT judge ---
    if not args.dry_run:
        if not os.getenv("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY not set. Use --dry_run for auto metrics only.",
                  file=sys.stderr)
            sys.exit(2)

        client = OpenAI(
            base_url=os.getenv("BASE_URL") or os.getenv("OPENAI_BASE_URL") or None
        )
        print(f"Running GPT dual judge ({args.model}, workers={args.workers}) ...",
              file=sys.stderr)

        def _judge_task(idx_entry):
            idx, entry = idx_entry
            return idx, dual_judge_one(client, args.model, entry)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_judge_task, (i, entry)): i
                for i, entry in enumerate(data)
            }
            for fut in tqdm(as_completed(futures), total=len(data), desc="GPT judge"):
                idx, judge_result = fut.result()
                results[idx]["judge"] = judge_result
    else:
        print("Dry run — skipping GPT judge.", file=sys.stderr)

    # --- aggregate ---
    report = aggregate_report(results)
    print_report(report)

    # --- save ---
    out_path = args.output_json or str(
        Path(args.input_json).parent / "audit_results.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {"report": report, "per_sample": results}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
