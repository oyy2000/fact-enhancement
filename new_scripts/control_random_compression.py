"""
Control Experiment: Random Step Compression
Creates randomly compressed reasoning traces (no semantic understanding)
to test whether DenseSteer's gains come from the dense rewriting or
from generic structural changes.

Pipeline:
1. Load the model's original correct samples (EM=1)
2. Create "randomly compressed" versions by randomly merging adjacent lines
3. Save as paired data in the same format as gpt_rewrites
4. Extract steering vectors using the same pipeline
5. Evaluate on GSM8K
"""
import json
import random
import os
import re
import argparse
from pathlib import Path
from copy import deepcopy

BASE = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement"


def random_compress(text: str, merge_ratio: float = 0.3, seed: int = 42) -> str:
    """Randomly merge adjacent lines to simulate compression without semantic understanding.
    
    merge_ratio: fraction of mergeable line boundaries to merge
    """
    rng = random.Random(seed)
    lines = text.split('\n')
    if len(lines) <= 2:
        return text

    merged = [lines[0]]
    for i in range(1, len(lines)):
        line = lines[i]
        if (line.strip() and merged[-1].strip() and
            rng.random() < merge_ratio and
            not re.match(r'^#{1,4}\s', line) and
            not re.match(r'^\*\*Step', line) and
            not re.match(r'^\d+\.', line) and
            not line.strip().startswith('\\[') and
            not line.strip().startswith('$$')):
            merged[-1] = merged[-1].rstrip() + ' ' + line.lstrip()
        else:
            merged.append(line)

    return '\n'.join(merged)


def shuffle_compress(text: str, seed: int = 42) -> str:
    """Shuffle the order of words within each sentence while keeping
    the sentence boundaries and mathematical expressions intact.
    This destroys semantic meaning while preserving rough structure."""
    rng = random.Random(seed)
    lines = text.split('\n')
    result = []
    for line in lines:
        if not line.strip():
            result.append(line)
            continue
        parts = re.split(r'(\\\(.*?\\\)|\\\[.*?\\\]|\$\$.*?\$\$|\$.*?\$|<<.*?>>)', line)
        new_parts = []
        for part in parts:
            if re.match(r'^\\\(|\\\[|\$\$|\$|<<', part):
                new_parts.append(part)
            else:
                words = part.split()
                if len(words) > 3:
                    mid = words[1:-1]
                    rng.shuffle(mid)
                    words = [words[0]] + mid + [words[-1]]
                new_parts.append(' '.join(words))
        result.append(''.join(new_parts))
    return '\n'.join(result)


def create_random_compression_pairs(source_path: str, output_path: str,
                                     num_examples: int = 50,
                                     compression_type: str = "random_merge"):
    """Create control experiment data with random compression."""
    data = json.load(open(source_path))
    
    em1_samples = [ex for ex in data if ex.get("exact_match", 0) == 1.0
                   and len(ex.get("resp_before", "")) > 50]
    
    em1_samples = em1_samples[-num_examples:]
    
    print(f"Using {len(em1_samples)} EM=1 samples from {source_path}")

    output_data = []
    for i, ex in enumerate(em1_samples):
        new_ex = deepcopy(ex)
        original = ex["resp_before"]
        
        if compression_type == "random_merge":
            compressed = random_compress(original, merge_ratio=0.4, seed=42 + i)
        elif compression_type == "shuffle":
            compressed = shuffle_compress(original, seed=42 + i)
        else:
            raise ValueError(f"Unknown compression_type: {compression_type}")

        new_ex["resp_after"] = compressed
        new_ex["resp_rewrite_style"] = f"control_{compression_type}"
        new_ex["resp_rewrite_ok"] = True
        output_data.append(new_ex)
        
        if i < 2:
            orig_lines = len(original.split('\n'))
            comp_lines = len(compressed.split('\n'))
            print(f"  Sample {i}: lines {orig_lines} -> {comp_lines}, "
                  f"chars {len(original)} -> {len(compressed)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(output_data)} samples to {output_path}")
    return output_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen_Qwen2.5-3B-Instruct",
                        help="Model folder name (underscore format)")
    parser.add_argument("--compression", default="random_merge",
                        choices=["random_merge", "shuffle"])
    parser.add_argument("--num_examples", type=int, default=50)
    args = parser.parse_args()

    source_path = os.path.join(BASE, "gpt_rewrites_unified_new",
                                args.model, "rewritten_old.json")
    
    output_dir = os.path.join(BASE, "control_experiments", args.model)
    output_path = os.path.join(output_dir,
                                f"rewritten_control_{args.compression}.json")

    create_random_compression_pairs(
        source_path, output_path,
        num_examples=args.num_examples,
        compression_type=args.compression,
    )


if __name__ == "__main__":
    main()
