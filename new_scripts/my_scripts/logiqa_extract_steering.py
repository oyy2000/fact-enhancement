#!/usr/bin/env python3
"""
Extract steering vectors from LogiQA or HotpotQA contrastive pairs (resp_after = dense, resp_before = sparse).

Input: JSON from 00_gpt_modification.py (list of dicts with resp_before, resp_after, doc, exact_match)
       OR JSONL merged to list.

Uses the same train_steering_vector API as 01_extract_vectors_large.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from utils import steering_prompt_from_doc

from steering_vectors import train_steering_vector


def load_samples(path: str):
    path = str(path)
    if path.endswith(".jsonl"):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        for k in ["samples", "instances", "data"]:
            if k in obj and isinstance(obj[k], list):
                return obj[k]
    return obj


def get_exact_match(ex: dict) -> float | None:
    if "exact_match" in ex:
        try:
            return float(ex["exact_match"])
        except Exception:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--in_path", required=True, help="rewritten_old.json or JSONL")
    ap.add_argument("--out_dir", default="", help="default: exps/logiqa_densesteer/vectors/...")
    ap.add_argument(
        "--tag",
        default="dense_gpt",
        help="Subfolder name to avoid overwriting (e.g. dense_gpt, infamily_7b).",
    )
    ap.add_argument("--num_examples", type=int, default=50)
    ap.add_argument("--layers", default="6", help="Comma-separated layer indices")
    ap.add_argument(
        "--domain",
        default="",
        choices=["", "logiqa", "hotpotqa"],
        help="Subdir under exps/ for vectors (default: infer from first sample's filter).",
    )
    args = ap.parse_args()

    layer_list = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    base = Path(__file__).resolve().parents[2]
    model_tag = args.model.replace("/", "_") + "_applied"

    domain = args.domain
    if not domain:
        samples_preview = load_samples(args.in_path)
        if samples_preview:
            d0 = (samples_preview[0].get("doc") or {})
            flt = samples_preview[0].get("filter") or d0.get("filter")
            domain = "hotpotqa" if flt == "hotpotqa_cot" else "logiqa"
        else:
            domain = "logiqa"

    parent_dir = "logiqa_densesteer" if domain == "logiqa" else "hotpotqa_densesteer"
    out_root = (
        Path(args.out_dir)
        if args.out_dir
        else base
        / "exps"
        / parent_dir
        / "vectors"
        / args.model.replace("/", "_")
        / f"N{args.num_examples}_{args.tag}"
    )
    model_out_dir = out_root / model_tag
    model_out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.in_path)
    by_id = {ex["doc_id"]: ex for ex in samples if "doc_id" in ex}
    doc_ids = sorted(by_id.keys())

    selected_ids = []
    for did in reversed(doc_ids):
        if get_exact_match(by_id[did]) == 1.0:
            rb = (by_id[did].get("resp_before") or "").strip()
            ra = (by_id[did].get("resp_after") or "").strip()
            if rb and ra and rb != ra:
                selected_ids.append(did)
                if len(selected_ids) >= args.num_examples:
                    break
    selected_ids = list(reversed(selected_ids))
    print(f"Selected {len(selected_ids)} contrastive pairs for steering.", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    training_samples = []
    for did in selected_ids:
        ex = by_id[did]
        doc = ex["doc"]
        prompt = steering_prompt_from_doc(doc, ex)
        training_samples.append((prompt + ex["resp_after"], prompt + ex["resp_before"]))

    if not training_samples:
        print("No training samples; check resp_before/resp_after and exact_match.", flush=True)
        sys.exit(1)

    steering_vector = train_steering_vector(
        model=model,
        tokenizer=tokenizer,
        training_samples=training_samples,
        layers=layer_list,
        layer_type="decoder_block",
        move_to_cpu=True,
        read_token_index=-1,
        show_progress=True,
        batch_size=1,
    )

    save_path = model_out_dir / "steering_vector.pt"
    torch.save(steering_vector, save_path)
    print(f"Saved {save_path}", flush=True)

    norms = {}
    for layer_idx, vec in steering_vector.layer_activations.items():
        norms[int(layer_idx)] = float(vec.norm().item())
    with open(model_out_dir / "vector_norms.json", "w") as f:
        json.dump(norms, f, indent=2)


if __name__ == "__main__":
    main()
