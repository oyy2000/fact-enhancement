import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def summarize(values: List[float]) -> Tuple[int, float]:
    count = len(values)
    accuracy = sum(values) / count if count else float("nan")
    return count, accuracy


def compute_metrics(path: Path, limit_per_filter: int = 1000) -> Dict[str, Dict[str, float]]:
    strict: List[float] = []
    flexible: List[float] = []

    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            em = obj.get("exact_match")
            filt = obj.get("filter")

            if filt == "strict-match" and len(strict) < limit_per_filter:
                strict.append(em)
            elif filt == "flexible-extract" and len(flexible) < limit_per_filter:
                flexible.append(em)

            if len(strict) >= limit_per_filter and len(flexible) >= limit_per_filter:
                break

    n_strict, acc_strict = summarize(strict)
    n_flexible, acc_flexible = summarize(flexible)

    return {
        "strict_match": {"count": n_strict, "accuracy": acc_strict},
        "flexible_extract": {"count": n_flexible, "accuracy": acc_flexible},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute accuracy for strict-match and flexible-extract filters.")
    parser.add_argument("--limit", type=int, default=1000, help="Max samples per filter (default: 1000)")
    args = parser.parse_args()
    args.jsonl = Path("/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/long_cot_vs_short_cot/UWNSL__Qwen2.5-3B-Instruct_Mix-Long/samples_gsm8k_cot_zeroshot_unified_2026-01-28T11-47-32.581025.jsonl")
    metrics = compute_metrics(args.jsonl, args.limit)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
