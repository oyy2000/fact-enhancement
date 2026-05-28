#!/usr/bin/env python3
"""Pair LogiQA CoT outputs from small vs large model for InFamilySteer (same as 00_large_model_selection)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def get_resps_0_0(obj):
    resps = obj.get("resps")
    if resps and isinstance(resps, list) and len(resps) > 0:
        if isinstance(resps[0], list) and len(resps[0]) > 0:
            return resps[0][0], True
        if isinstance(resps[0], str):
            return resps[0], True
    return None, False


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small_jsonl", required=True)
    ap.add_argument("--large_jsonl", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--small_model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--large_model", default="Qwen/Qwen2.5-7B-Instruct")
    args = ap.parse_args()

    small_rows = load_jsonl(args.small_jsonl)
    large_rows = load_jsonl(args.large_jsonl)
    ref_map = {}
    for r in large_rows:
        did = r.get("doc_id")
        resp, ok = get_resps_0_0(r)
        if ok and did is not None:
            ref_map[did] = resp

    paired = []
    for row in small_rows:
        did = row.get("doc_id")
        rb, ok = get_resps_0_0(row)
        ra = ref_map.get(did)
        row = dict(row)
        row["resp_before"] = rb if ok else ""
        row["resp_after"] = ra if ra else ""
        row["resp_source_model"] = args.small_model
        row["resp_target_model"] = args.large_model
        paired.append(row)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(paired, f, indent=2, ensure_ascii=False)
    print(f"Paired {len(paired)} rows -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
