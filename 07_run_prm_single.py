import argparse
import json
from prm_eval_core import run_dataset

# ============================================================
# ARGPARSE
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", required=True, help="Logical model name (for JSON key)")
parser.add_argument("--gen_model", required=True, help="HF repo id for decoder tokenizer")
parser.add_argument("--layer", required=True)
parser.add_argument("--lam", required=True)
parser.add_argument("--jsonl", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--eval_start", type=int, default=100)
args = parser.parse_args()

# ============================================================
# RUN PRM EVAL (RECORD ONLY)
# ============================================================

Y, STEP_SCORES_ALL, STEP_TEXTS_ALL, STEP_TOKEN_LEN = run_dataset(
    jsonl_path=args.jsonl,
    gen_model_name=args.gen_model,
    eval_start=args.eval_start
)

# ============================================================
# SAVE RESULT (NO METRICS, ONLY FACTS)
# ============================================================

res = {
    args.model_name: {
        args.layer: {
            args.lam: {
                "file_used": args.jsonl,
                "gen_model": args.gen_model,

                # raw records only
                "Y": Y,
                "step_scores": STEP_SCORES_ALL,
                "step_texts": STEP_TEXTS_ALL,
                "step_token_len": STEP_TOKEN_LEN
            }
        }
    }
}

with open(args.out, "w") as f:
    json.dump(res, f, indent=2)

print(f"✔ Saved chunk → {args.out}")
