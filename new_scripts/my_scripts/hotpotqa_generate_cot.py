#!/usr/bin/env python3
"""
Generate chain-of-thought HotpotQA (LongBench) responses for DenseSteer / InFamilySteer.

Uses THUDM/LongBench hotpotqa. If no train split exists, uses test (note: then E12 HotpotQA
eval overlaps calibration data — prefer LogiQA when possible).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import DatasetDict, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from utils import hotpotqa_format_problem_cot, hotpotqa_qwen_chat_prompt, hotpotqa_answer_match


def pick_split(ds_dict: DatasetDict, preferred: str) -> tuple[str, object]:
    if preferred in ds_dict and len(ds_dict[preferred]) > 0:
        return preferred, ds_dict[preferred]
    for name in ("train", "validation", "test"):
        if name in ds_dict and len(ds_dict[name]) > 0:
            print(f"[warn] split '{preferred}' unavailable; using '{name}'", flush=True)
            return name, ds_dict[name]
    raise RuntimeError(f"No non-empty split in dataset keys={list(ds_dict.keys())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--split", default="train", help="Preferred split name (fallback: train→val→test)")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument(
        "--out_jsonl",
        default="",
        help="Default: exps/hotpotqa_densesteer/samples/<model>_hotpotqa_cot_<split>.jsonl",
    )
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--device_map", default="auto")
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[2]
    model_tag = args.model.replace("/", "_")

    print("Loading THUDM/LongBench hotpotqa...", flush=True)
    ds_dict = load_dataset("THUDM/LongBench", "hotpotqa", trust_remote_code=True)
    split_name, ds = pick_split(ds_dict, args.split)
    n = min(args.limit, len(ds))
    ds = ds.select(range(n))

    out_jsonl = (
        Path(args.out_jsonl)
        if args.out_jsonl
        else base
        / "exps"
        / "hotpotqa_densesteer"
        / "samples"
        / f"{model_tag}_hotpotqa_cot_{split_name}.jsonl"
    )
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    ).eval()

    fout = open(out_jsonl, "w", encoding="utf-8")
    n_ok = 0
    for i in tqdm(range(len(ds)), desc="hotpotqa_generate"):
        row = ds[i]
        context = row["context"]
        q = row["input"]
        answers = row.get("answers", row.get("answer", []))

        prompt = hotpotqa_qwen_chat_prompt(context, q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        ok = hotpotqa_answer_match(gen, answers)
        em = 1.0 if ok else 0.0
        if ok:
            n_ok += 1

        obj = {
            "doc_id": i,
            "doc": {
                "context": context,
                "question": hotpotqa_format_problem_cot(context, q),
                "question_stem": q,
                "answers": answers,
            },
            "resps": [[gen]],
            "exact_match": em,
            "filter": "hotpotqa_cot",
            "task": "hotpotqa_cot",
            "model": args.model,
            "split": split_name,
        }
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    fout.close()
    print(f"Wrote {n} lines to {out_jsonl}", flush=True)
    print(f"Lenient exact-match count: {n_ok} / {n}", flush=True)


if __name__ == "__main__":
    main()
