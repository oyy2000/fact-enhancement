#!/usr/bin/env python3
"""
Compute teacher-forcing NLL on *generated* continuations for E0–E3.2 control runs.

Uses lm_eval sample jsonl: prompt from arguments.gen_args_0.arg_0, text from resps[0][0].
Labels mask the prompt; loss is mean cross-entropy over generated tokens only (nats).

Also reports joint metric (same ρ as Sec. 3.2 / fill_e0_e3_control_tokens_rho: steps
split by \\n\\n, ρ = total tokenizer tokens / n_steps):
  NLL_total_i = nll_mean_i * n_gen_tokens_i  (sum of -log p over generated tokens),
  J_i = ρ_i * exp(-NLL_total_i),  J = (1/N) * sum_i J_i.

Default: pick GSM8K best flexible-extract sweep job per experiment (same rule as
fill_e0_e3_control_tokens_rho.py).

Examples:
  python control_generation_nll.py --experiment e0 --gpu 0
  python control_generation_nll.py --samples-jsonl /path/to/samples_....jsonl --out-json summary.json
  python control_generation_nll.py --experiment e1 --shard 0 --num-shards 4
  python control_generation_nll.py --experiment e0 --only-correct
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path("/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement")
CONTROL_MODEL_DIR = BASE / "control_experiments" / "Qwen_Qwen2.5-3B-Instruct"
SWEEP_DIR = CONTROL_MODEL_DIR / "sweep_eval"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

EXPERIMENT_PREFIXES = ("e0", "e1", "e2", "e3", "e3.2")


def is_correct_gsm8k(rec: dict) -> bool:
    """lm_eval gsm8k sample line: exact_match 1.0 iff flexible-extract match."""
    try:
        return float(rec.get("exact_match", 0)) >= 1.0 - 1e-9
    except (TypeError, ValueError):
        return False


def latest_results(jobdir: Path) -> Path | None:
    js = list(jobdir.glob("**/results_*.json"))
    return max(js, key=lambda p: p.stat().st_mtime) if js else None


def samples_for_result(results_path: Path) -> Path | None:
    ts = results_path.stem.replace("results_", "")
    parent = results_path.parent
    cand = list(parent.glob(f"samples_*_{ts}.jsonl"))
    if cand:
        return cand[0]
    alls = list(parent.glob("samples_*.jsonl"))
    return max(alls, key=lambda p: p.stat().st_mtime) if alls else None


def best_sweep_samples(exp_prefix: str) -> tuple[Path | None, float | None, str | None]:
    best_acc = -1.0
    best_rp: Path | None = None
    best_dir_name: str | None = None
    for d in sorted(SWEEP_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith(exp_prefix + "_L6_lam"):
            continue
        rp = latest_results(d)
        if rp is None:
            continue
        data = json.loads(rp.read_text(encoding="utf-8"))
        acc = data["results"]["gsm8k_cot_zeroshot_unified"]["exact_match,flexible-extract"]
        if acc > best_acc:
            best_acc, best_rp, best_dir_name = acc, rp, d.name
    sp = samples_for_result(best_rp) if best_rp else None
    return sp, (best_acc if best_rp else None), best_dir_name


def extract_prompt_from_record(rec: dict) -> str | None:
    args = rec.get("arguments") or {}
    g0 = args.get("gen_args_0") or {}
    p = g0.get("arg_0")
    return p if isinstance(p, str) and p.strip() else None


def extract_generation(rec: dict) -> str:
    rs = rec.get("resps") or []
    if not rs or not rs[0]:
        return ""
    t = rs[0][0]
    return t if isinstance(t, str) else ""


def count_tokens_rho(text: str, tokenizer) -> tuple[int, float, float]:
    """Match fill_e0_e3_control_tokens_rho.count_tokens_rho."""
    steps = [s.strip() for s in (text or "").split("\n\n") if s.strip()]
    if not steps:
        return 1, 0.0, 0.0
    n = len(steps)
    lens = [len(tokenizer.encode(s, add_special_tokens=False)) for s in steps]
    tot = float(sum(lens))
    return n, tot, tot / n


def joint_J_i(rho: float, nll_mean: float, n_gen_tokens: int) -> float:
    """J_i = rho * exp(-NLL_total); NLL_total = mean CE * T (sequence log-prob up to constant)."""
    if n_gen_tokens <= 0 or rho < 0:
        return 0.0
    nll_total = float(nll_mean) * float(n_gen_tokens)
    # exp(-nll_total) underflows to 0.0 for very long / unlikely sequences
    return float(rho) * math.exp(-nll_total)


def nll_one(
    prompt: str,
    continuation: str,
    model,
    tokenizer,
    max_length: int,
) -> tuple[float | None, int]:
    if not continuation.strip():
        return None, 0

    full_text = prompt + continuation
    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(model.device)

    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    prompt_len = min(int(prompt_ids.shape[1]), seq_len)

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    if torch.all(labels == -100):
        return None, 0

    n_gen = int((labels != -100).sum().item())
    with torch.inference_mode():
        out = model(**inputs, labels=labels)
    loss = out.loss
    if loss is None or torch.isnan(loss) or torch.isinf(loss):
        return None, n_gen
    return float(loss.item()), n_gen


def load_records_dedup(path: Path) -> list[dict]:
    by_doc: dict = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_doc[rec.get("doc_id")] = rec
    return list(by_doc.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=EXPERIMENT_PREFIXES, default=None)
    parser.add_argument("--samples-jsonl", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-per-doc", type=Path, default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-docs", type=int, default=None, help="Score only first N docs after dedup (debug).")
    parser.add_argument(
        "--only-correct",
        action="store_true",
        help="Restrict to samples with exact_match==1 (GSM8K correct under lm_eval).",
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    args = parser.parse_args()

    if args.samples_jsonl is None:
        if args.experiment is None:
            print("Need --experiment or --samples-jsonl", file=sys.stderr)
            sys.exit(1)
        sp, acc, sweep_name = best_sweep_samples(args.experiment)
        if sp is None or not sp.is_file():
            print(f"No samples found for experiment {args.experiment}", file=sys.stderr)
            sys.exit(1)
        samples_path = sp
        meta = {
            "experiment": args.experiment,
            "best_sweep_dir": sweep_name,
            "best_flexible_extract_acc": acc,
        }
    else:
        samples_path = args.samples_jsonl
        if not samples_path.is_file():
            print(f"Missing file {samples_path}", file=sys.stderr)
            sys.exit(1)
        meta = {"samples_jsonl": str(samples_path)}

    records = load_records_dedup(samples_path)
    n_deduped = len(records)
    n_correct_pool: int | None = None
    if args.only_correct:
        records = [r for r in records if is_correct_gsm8k(r)]
        n_correct_pool = len(records)
        meta["only_correct"] = True
        meta["n_docs_deduped_before_correct_filter"] = n_deduped
        meta["n_correct_pool"] = n_correct_pool
    if args.max_docs is not None:
        records = records[: args.max_docs]
    if args.num_shards > 1:
        records = [r for i, r in enumerate(records) if i % args.num_shards == args.shard]

    if not records:
        print("No samples left after filters (dedup / --only-correct / --max-docs / shard).", file=sys.stderr)
        sys.exit(1)

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    nlls: list[float] = []
    n_toks: list[int] = []
    j_values: list[float] = []
    n_skip = 0
    per_doc_rows: list[dict] = []

    for rec in tqdm(records, desc="nll"):
        prompt = extract_prompt_from_record(rec)
        gen = extract_generation(rec)
        if prompt is None:
            n_skip += 1
            per_doc_rows.append(
                {
                    "doc_id": rec.get("doc_id"),
                    "exact_match": rec.get("exact_match"),
                    "nll_mean": None,
                    "n_gen_tokens": 0,
                    "nll_total": None,
                    "rho": None,
                    "J_i": None,
                    "skip": "no_prompt",
                }
            )
            continue
        nm, nt = nll_one(prompt, gen, model, tokenizer, args.max_length)
        if nm is None:
            n_skip += 1
            per_doc_rows.append(
                {
                    "doc_id": rec.get("doc_id"),
                    "exact_match": rec.get("exact_match"),
                    "nll_mean": None,
                    "n_gen_tokens": nt,
                    "nll_total": None,
                    "rho": None,
                    "J_i": None,
                    "skip": "empty_or_invalid",
                }
            )
            continue
        nlls.append(nm)
        n_toks.append(nt)
        _, _, rho = count_tokens_rho(gen, tokenizer)
        nll_total = nm * nt
        ji = joint_J_i(rho, nm, nt)
        j_values.append(ji)
        per_doc_rows.append(
            {
                "doc_id": rec.get("doc_id"),
                "exact_match": rec.get("exact_match"),
                "nll_mean": nm,
                "n_gen_tokens": nt,
                "nll_total": nll_total,
                "rho": rho,
                "J_i": ji,
                "skip": None,
            }
        )

    summary = {
        **meta,
        "model": args.model,
        "samples_jsonl": str(samples_path),
        "n_docs": len(records),
        "n_scored": len(nlls),
        "n_skipped": n_skip,
        "mean_nll_per_gen_token": sum(nlls) / len(nlls) if nlls else None,
        "weighted_mean_nll": (
            sum(n * t for n, t in zip(nlls, n_toks)) / sum(n_toks) if n_toks and sum(n_toks) > 0 else None
        ),
        "total_gen_tokens": sum(n_toks),
        "joint_metric_J": sum(j_values) / len(j_values) if j_values else None,
        "joint_metric_note": (
            "J = mean_i( rho_i * exp(-NLL_total_i) ); "
            "rho = sum_tok(step)/n_steps (blank-line steps); "
            "NLL_total_i = nll_mean_i * n_gen_tokens_i"
        ),
        "shard": args.shard,
        "num_shards": args.num_shards,
    }

    out_json = args.out_json
    if out_json is None:
        tag = args.experiment or samples_path.stem
        shard_sfx = f"_shard{args.shard}of{args.num_shards}" if args.num_shards > 1 else ""
        cor_sfx = "_correct_only" if args.only_correct else ""
        out_json = CONTROL_MODEL_DIR / f"nll_generation_{tag}{cor_sfx}{shard_sfx}.json"

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if args.out_per_doc:
        p = Path(args.out_per_doc)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for row in per_doc_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
