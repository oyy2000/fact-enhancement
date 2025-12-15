# run_prm_single.py
import argparse, json, numpy as np
from scipy.stats import pearsonr
from prm_eval_core import run_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model")
parser.add_argument("--layer")
parser.add_argument("--lam")
parser.add_argument("--jsonl")
parser.add_argument("--out")
args = parser.parse_args()

# Evaluate
F_full, F_no_last, F_hard, F_hard_no_last, EARLY, PREFIX, AVG_LEN, STEPS, Y, STEP_SCORES_ALL, STEP_TEXTS_ALL = run_dataset(args.jsonl)

EARLY_clean = [
    e if e is not None else (max([x for x in EARLY if x is not None])+1)
    for e in EARLY
]

res = {
    args.model: {
        args.layer: {
            args.lam: {
                "corr_full": float(pearsonr(F_full, Y)[0]),
                "corr_hard": float(pearsonr(F_hard, Y)[0]),
                "corr_avg_prefix": float(pearsonr(PREFIX, Y)[0]),
                "corr_avg_steps": float(pearsonr(STEPS, Y)[0]),
                "corr_avg_first_error": float(pearsonr(EARLY_clean, Y)[0]),
                
                "avg_prefix": float(np.mean(PREFIX)),
                "avg_first_error": float(np.mean([e for e in EARLY if e is not None])),
                "avg_steps": float(np.mean(STEPS)),
                "file_used": args.jsonl,
                "Y": Y,
                "step_scores": STEP_SCORES_ALL,
                "step_texts": STEP_TEXTS_ALL
            }
        }
    }
}



with open(args.out, "w") as f:
    json.dump(res, f, indent=2)

print(f"✔ Saved chunk → {args.out}")
