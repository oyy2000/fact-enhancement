#!/usr/bin/env python3
"""
Generate chain-of-thought LogiQA responses for DenseSteer / InFamilySteer calibration.

Uses EleutherAI/logiqa train split by default (avoid test leakage for vector construction).
Output JSONL lines compatible with 00_gpt_modification.py (resps, doc_id, doc, exact_match).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from utils import logiqa_format_problem, logiqa_qwen_chat_prompt


def parse_final_answer(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"Final Answer:\s*([A-Da-d])\b", text, re.I)
    if m:
        return m.group(1).upper()
    for pat in [
        r"(?:correct\s+)?(?:answer|option)\s+is\s*:?\s*\*?\s*([A-Da-d])\b",
        r"\b(?:choose|select|pick)\s+(?:option\s+)?([A-Da-d])\b",
        r"\b(?:Therefore|Thus|Hence),?\s+(?:the\s+)?(?:answer|correct\s+option)\s+is\s*:?\s*([A-Da-d])\b",
    ]:
        m2 = re.search(pat, text, re.I)
        if m2:
            return m2.group(1).upper()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for ln in reversed(lines[-8:]):
        m3 = re.match(r"^([A-Da-d])\s*[\.\)]?$", ln)
        if m3:
            return m3.group(1).upper()
    m4 = re.findall(r"\b([A-D])\b", text[-200:])
    if m4:
        return m4[-1].upper()
    return None


def label_to_letter(label) -> str:
    if isinstance(label, int):
        return chr(65 + int(label))
    s = str(label).strip().upper()
    if len(s) == 1 and s in "ABCD":
        return s
    if len(s) == 1 and s in "abcd":
        return s.upper()
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--split", default="train", choices=["train", "validation", "test"])
    ap.add_argument("--limit", type=int, default=800, help="Max examples to generate")
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument(
        "--out_jsonl",
        default="",
        help="Output path (default: exps/logiqa_densesteer/samples/<model_tag>_logiqa_cot.jsonl)",
    )
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--device_map", default="auto")
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[2]
    model_tag = args.model.replace("/", "_")
    out_jsonl = (
        Path(args.out_jsonl)
        if args.out_jsonl
        else base / "exps" / "logiqa_densesteer" / "samples" / f"{model_tag}_logiqa_cot_{args.split}.jsonl"
    )
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset EleutherAI/logiqa split={args.split}...", flush=True)
    ds = load_dataset("EleutherAI/logiqa", "logiqa", split=args.split, trust_remote_code=True)
    n = min(args.limit, len(ds))
    ds = ds.select(range(n))

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
    for i in tqdm(range(len(ds)), desc="generate"):
        row = ds[i]
        context = row["context"]
        question = row["question"]
        options = list(row["options"])
        gold = label_to_letter(row["label"])

        prompt = logiqa_qwen_chat_prompt(context, question, options)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        pred = parse_final_answer(gen)
        em = 1.0 if pred is not None and pred == gold else 0.0
        if em >= 1.0:
            n_ok += 1

        obj = {
            "doc_id": i,
            "doc": {
                "context": context,
                "question": logiqa_format_problem(context, question, options),
                "question_stem": question,
                "options": options,
                "gold": gold,
            },
            "resps": [[gen]],
            "exact_match": em,
            "filter": "logiqa_cot",
            "task": "logiqa_cot",
            "model": args.model,
            "split": args.split,
        }
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    fout.close()
    print(f"Wrote {n} lines to {out_jsonl}", flush=True)
    print(f"Exact-match (letter) count: {n_ok} / {n}", flush=True)


if __name__ == "__main__":
    main()
