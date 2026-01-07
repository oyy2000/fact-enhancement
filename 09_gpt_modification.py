#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rewrite lm-eval-harness JSONL samples by rewriting ONLY obj["resps"][0][0]
(the model output), for the LAST N examples.

- No final answer requirement.
- No fallback-on-mismatch (we don't check mismatch at all).
- Optionally overwrite obj["resps"][0][0] if --overwrite_resps is set.

Input JSONL line schema (typical):
  {
    "doc_id": ...,
    "doc": {"question": ..., "answer": ...},
    "target": "...",
    "resps": [[ "...model output..." ]],
    ...
  }

Outputs:
  OUT_JSONL: same JSONL with added fields:
    - resp_before: original resps[0][0] (always preserved if exists)
    - resp_after: rewritten resps[0][0] (or original if not rewritten)
    - resp_rewrite_applied: True/False
    - resp_rewrite_ok: True/False/None   (None for passthrough)
    - resp_rewrite_reason: short string
    - rewrite_index: integer index in file (for debugging)

  OUT_JSON: pretty JSON array containing ONLY rewritten-range samples
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

from tqdm import tqdm
from openai import OpenAI  # pip install openai


# -----------------------------
# Defaults (your paths)
# -----------------------------
IN_JSONL = "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_qwen_familiy/gsm8k_cot_zeroshot/Qwen2.5-1.5B-Instruct_L8_BASELINE/Qwen__Qwen2.5-1.5B-Instruct/samples_gsm8k_cot_zeroshot_2025-12-11T02-46-20.289597.jsonl"
OUT_JSONL = "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_qwen_familiy/gsm8k_cot_zeroshot/Qwen2.5-1.5B-Instruct_L8_BASELINE/Qwen__Qwen2.5-1.5B-Instruct/samples_gsm8k_cot_zeroshot_rewritten.jsonl"


def get_resps_0_0(obj: Dict[str, Any]) -> Tuple[Optional[str], bool]:
    """
    Returns (text, exists_flag).
    exists_flag=True means resps[0][0] exists structurally (even if empty string).
    """
    resps = obj.get("resps")
    if isinstance(resps, list) and len(resps) > 0 and isinstance(resps[0], list) and len(resps[0]) > 0:
        return (resps[0][0], True)
    return (None, False)


def set_resps_0_0(obj: Dict[str, Any], new_text: str) -> bool:
    """
    Set obj["resps"][0][0] = new_text if structure exists.
    Returns True if set, False otherwise.
    """
    resps = obj.get("resps")
    if isinstance(resps, list) and len(resps) > 0 and isinstance(resps[0], list) and len(resps[0]) > 0:
        obj["resps"][0][0] = new_text
        return True
    return False

def build_prompt(question: str, original_resp: str) -> str:
    return f"""You will lightly rewrite the solution by CONSERVATIVELY merging steps, while keeping the SAME style and meaning.

Hard constraints:
- Keep the SAME meaning and do NOT change the final conclusion/answer implied by the solution.
- Do NOT invent new reasoning. Only compress/merge/rephrase existing steps.
- Keep the style and tone the SAME as the original (do not change formality, phrasing habits, or formatting conventions).
- Only merge steps when it is NECESSARY and safe (e.g., two adjacent lines that are clearly redundant or tightly coupled).
  Do NOT aggressively minimize the number of steps. If merging would change the “feel” or clarity, keep the original steps.
- When you merge, prefer merging 2 adjacent steps into 1 step (avoid merging many lines at once).
- Keep computations consistent with the original (same numbers/operations, no new math).
- Preserve special markers like "<<a=b>>" if they appear; do not introduce many new ones.
- Output plain text only. No bullet points or added commentary.

Question:
{question}

Original solution (model output):
{original_resp}

Now output ONLY the rewritten solution (same style, with a few necessary merges):
"""


def call_gpt_rewrite(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int = 220,
    temperature: float = 0.0,
    retries: int = 4,
    sleep_base: float = 1.5,
) -> str:
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            return (resp.output_text or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(sleep_base * (2 ** attempt))
    raise RuntimeError(f"OpenAI API failed after {retries} retries: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", default=IN_JSONL, help="Input JSONL file path")
    ap.add_argument("--out_jsonl", default=OUT_JSONL, help="Output JSONL file path")
    ap.add_argument("--out_json", default=None, help="Pretty JSON output path (default: out_jsonl->.json)")
    ap.add_argument("--model", default="gpt-5.1", help="OpenAI model name (Responses API)")
    ap.add_argument("--rewrite_last_n", type=int, default=1, help="Rewrite only the last N examples")
    ap.add_argument("--overwrite_resps", action="store_true", help="Overwrite obj['resps'][0][0] with rewritten text")
    ap.add_argument("--max_output_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between API calls")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY env var.", file=sys.stderr)
        sys.exit(2)

    out_json_path = args.out_json or args.out_jsonl.replace(".jsonl", ".json")

    # Read all non-empty lines
    lines = []
    with open(args.in_jsonl, "r", encoding="utf-8") as fin:
        for raw in fin:
            s = raw.strip()
            if s:
                lines.append(s)

    total = len(lines)
    if total == 0:
        print("Empty input.", file=sys.stderr)
        with open(args.out_jsonl, "w", encoding="utf-8") as fout:
            pass
        with open(out_json_path, "w", encoding="utf-8") as fjson:
            json.dump([], fjson, ensure_ascii=False, indent=2)
        return

    start_idx = max(0, total - args.rewrite_last_n)

    client = OpenAI()

    n_attempted = 0
    n_written = 0
    n_skipped_missing = 0

    rewritten_objs_for_json = []

    pbar = tqdm(total=total, desc="Rewrite resps[0][0]", unit="sample", dynamic_ncols=True)

    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for i, line in enumerate(lines):
            obj = json.loads(line)
            obj["rewrite_index"] = i

            resp00, exists = get_resps_0_0(obj)
            if exists:
                obj["resp_before"] = resp00 if resp00 is not None else ""
            else:
                obj["resp_before"] = None

            # passthrough
            if i < start_idx:
                obj["resp_rewrite_applied"] = False
                obj["resp_after"] = obj["resp_before"]
                obj["resp_rewrite_ok"] = None
                obj["resp_rewrite_reason"] = "not_in_last_n"

                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                pbar.update(1)
                continue

            # rewrite range
            obj["resp_rewrite_applied"] = True
            n_attempted += 1

            question = (obj.get("doc") or {}).get("question", "") or ""
            source_text = (resp00 or "").strip() if isinstance(resp00, str) else ""

            if (not exists) or (not question) or (not source_text):
                # 你说不需要 fallback；这里我们只是“无法改写”的情况直接原样输出并标记原因
                obj["resp_after"] = obj["resp_before"]
                obj["resp_rewrite_ok"] = False
                obj["resp_rewrite_reason"] = "missing_resps00_or_question_or_text"
                n_skipped_missing += 1

                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                rewritten_objs_for_json.append(obj)

                pbar.set_postfix_str(f"attempted={n_attempted} written={n_written} skipped={n_skipped_missing}")
                pbar.update(1)
                continue

            prompt = build_prompt(question, source_text)
            rewritten = call_gpt_rewrite(
                client=client,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_output_tokens,
                temperature=args.temperature,
            ).strip()

            obj["resp_after"] = rewritten
            obj["resp_rewrite_ok"] = True
            obj["resp_rewrite_reason"] = "ok"
            n_written += 1

            if args.overwrite_resps:
                set_ok = set_resps_0_0(obj, rewritten)
                if not set_ok:
                    obj["resp_rewrite_ok"] = False
                    obj["resp_rewrite_reason"] = "structure_missing_when_overwrite"

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            rewritten_objs_for_json.append(obj)

            if args.sleep > 0:
                time.sleep(args.sleep)

            pbar.set_postfix_str(f"attempted={n_attempted} written={n_written} skipped={n_skipped_missing}")
            pbar.update(1)

    pbar.close()

    # Pretty JSON: ONLY rewritten-range samples
    with open(out_json_path, "w", encoding="utf-8") as fjson:
        json.dump(rewritten_objs_for_json, fjson, ensure_ascii=False, indent=2)

    print(
        f"Done. total={total}, rewritten_range=[{start_idx},{total-1}], "
        f"attempted={n_attempted}, written={n_written}, skipped_missing={n_skipped_missing}, "
        f"pretty_json={out_json_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
