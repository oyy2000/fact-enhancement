#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to pair responses from two models into a single JSON dataset.
- resp_before: Output from TARGET_MODEL (Qwen 1.5B)
- resp_after:  Output from QWEN_14B_MODEL (Qwen 7B)
"""

import argparse
import json
import os
import sys
from tqdm import tqdm

# -----------------------------
# Defaults
# -----------------------------
REWRITE_FOLDER = "large_model_rewrites_unified_new"

# Note: Variable names preserved from original script context


QWEN_7B_MODEL_SAMPLES_PATH = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/no_vector/gsm8k_cot_zeroshot_unified/Qwen2.5-7B-Instruct_no_vector/Qwen__Qwen2.5-7B-Instruct/samples_gsm8k_cot_zeroshot_unified_2026-01-21T11-34-14.746371.jsonl"
QWEN_3B_MODEL_SAMPLES_PATH = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/no_vector/gsm8k_cot_zeroshot_unified/Qwen2.5-3B-Instruct_no_vector/Qwen__Qwen2.5-3B-Instruct/samples_gsm8k_cot_zeroshot_unified_2026-01-21T11-33-55.190404.jsonl"
LLAMA_3_1_8B_MODEL_SAMPLE_PATH = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/no_vector/gsm8k_cot_zeroshot_unified/Llama-3.1-8B-Instruct_no_vector/meta-llama__Llama-3.1-8B-Instruct/samples_gsm8k_cot_zeroshot_unified_2026-01-25T15-06-51.949587.jsonl"
LLAMA_3_2_1B_MODEL_SAMPLE_PATH = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/no_vector/gsm8k_cot_zeroshot_unified/Llama-3.2-1B-Instruct_no_vector/meta-llama__Llama-3.2-1B-Instruct/samples_gsm8k_cot_zeroshot_unified_2026-01-21T11-44-34.344707.jsonl"
LLAMA_3_2_3B_MODEL_SAMPLE_PATH = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/no_vector/gsm8k_cot_zeroshot_unified/Llama-3.2-3B-Instruct_no_vector/meta-llama__Llama-3.2-3B-Instruct/samples_gsm8k_cot_zeroshot_unified_2026-01-21T11-56-55.533987.jsonl"
# QWEN_14B_MODEL_SAMPLES_PATH = 

TARGET_MODEL_JSONL = LLAMA_3_2_1B_MODEL_SAMPLE_PATH
REF_MODEL_JSONL = LLAMA_3_1_8B_MODEL_SAMPLE_PATH
TARGET_MODEL = "meta-llama/Llama-3.2-1B-Instruct" #"Qwen/Qwen2.5-1.5B-Instruct"
REWRITE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def get_resps_0_0(obj):
    """
    Extracts the first response (sample 0) from the 'resps' list.
    Handles lm-eval structure which is usually [[resp1], [resp2], ...] or similar.
    """
    resps = obj.get("resps")
    if resps and isinstance(resps, list) and len(resps) > 0:
        # If list of lists (common in recent lm-eval)
        if isinstance(resps[0], list) and len(resps[0]) > 0:
             return resps[0][0], True
        # If flattened list of strings
        elif isinstance(resps[0], str):
             return resps[0], True
    return None, False

def load_jsonl(path):
    data = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    except Exception as e:
        print(f"Error loading {path}: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", default=TARGET_MODEL_JSONL, help="Input JSONL file path (resp_before source)")
    ap.add_argument("--ref_jsonl", default=REF_MODEL_JSONL, help="Reference JSONL file path (resp_after source)")
    
    # Kept for compatibility if user scripts pass it, though unused now
    ap.add_argument("--overwrite_resps", action="store_true") 
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--prompt_style", type=str, default="expert_leap") 

    args = ap.parse_args()
    
    # Determine Output Path
    out_path = os.path.join(REWRITE_FOLDER, TARGET_MODEL.replace("/", "_"))
    print(f"Target Output: {out_path}")
    os.makedirs(out_path, exist_ok=True)
    
    # 1. Load Reference Data (The 'After' / Ground Truth / Larger Model)
    print(f"Loading Reference (resp_after): {args.ref_jsonl}")
    ref_rows = load_jsonl(args.ref_jsonl)
    
    # Map doc_id -> response text
    ref_map = {}
    for r in ref_rows:
        did = r.get("doc_id")
        resp, exists = get_resps_0_0(r)
        if exists and did is not None:
            ref_map[did] = resp

    print(f"Loaded {len(ref_map)} unique reference responses.")

    # 2. Load Target Data (The 'Before' / Smaller Model)
    print(f"Loading Target (resp_before): {args.in_jsonl}")
    target_rows = load_jsonl(args.in_jsonl)

    # 3. Construct Paired Data
    paired_data = []
    
    for row in tqdm(target_rows, desc="Pairing"):
        doc_id = row.get("doc_id")
        
        # Get resp_before
        resp_before, before_exists = get_resps_0_0(row)
        
        # Get resp_after
        resp_after = ref_map.get(doc_id)
        
        # Update row with new fields
        row["resp_before"] = resp_before if before_exists else ""
        row["resp_after"] = resp_after if resp_after else ""
        
        # Optional: Add metadata about the pair
        row["resp_source_model"] = TARGET_MODEL  # Based on path
        row["resp_target_model"] = REWRITE_MODEL # Based on path
        
        paired_data.append(row)

    # 4. Save
    print(f"Saving {len(paired_data)} paired items to {out_path}")
    out_file = os.path.join(out_path, f"{REWRITE_MODEL.replace('/', '_')}_paired_responses.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(paired_data, f, indent=2, ensure_ascii=False)
        
    print("Done.")

if __name__ == "__main__":
    main()
