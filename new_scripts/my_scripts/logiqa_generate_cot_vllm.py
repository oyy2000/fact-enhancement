#!/usr/bin/env python3
"""
Fast LogiQA CoT generation using vLLM (batched).
Output format compatible with 00_gpt_modification.py and logiqa_extract_steering.py.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams

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
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    args = ap.parse_args()

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset EleutherAI/logiqa split={args.split}...", flush=True)
    ds = load_dataset("EleutherAI/logiqa", "logiqa", split=args.split, trust_remote_code=True)
    n = min(args.limit, len(ds))
    ds = ds.select(range(n))

    prompts = []
    metadata = []
    for i in range(len(ds)):
        row = ds[i]
        context = row["context"]
        question = row["question"]
        options = list(row["options"])
        gold = label_to_letter(row["label"])
        prompt = logiqa_qwen_chat_prompt(context, question, options)
        prompts.append(prompt)
        metadata.append({"context": context, "question": question, "options": options, "gold": gold, "idx": i})

    print(f"Loading vLLM model {args.model} (tp={args.tp})...", flush=True)
    llm = LLM(model=args.model, dtype="float16", tensor_parallel_size=args.tp,
              trust_remote_code=True, max_model_len=4096)
    sampling = SamplingParams(max_tokens=args.max_new_tokens, temperature=0, top_p=1.0)

    print(f"Generating {len(prompts)} samples...", flush=True)
    outputs = llm.generate(prompts, sampling)

    n_ok = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for out, meta in zip(outputs, metadata):
            gen = out.outputs[0].text
            pred = parse_final_answer(gen)
            em = 1.0 if pred is not None and pred == meta["gold"] else 0.0
            if em >= 1.0:
                n_ok += 1
            obj = {
                "doc_id": meta["idx"],
                "doc": {
                    "context": meta["context"],
                    "question": logiqa_format_problem(meta["context"], meta["question"], meta["options"]),
                    "question_stem": meta["question"],
                    "options": meta["options"],
                    "gold": meta["gold"],
                },
                "resps": [[gen]],
                "exact_match": em,
                "filter": "logiqa_cot",
                "task": "logiqa_cot",
                "model": args.model,
                "split": args.split,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote {n} lines to {out_path}", flush=True)
    print(f"Exact-match: {n_ok}/{n} = {n_ok/n*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
