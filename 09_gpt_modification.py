#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System Prompt:
Rewrite reasoning data to generate Steering Vector training pairs (Positive vs Negative).

Strategies available via --prompt_style:
[POSITIVE - For Layer 17 High Density]
1. expert_leap (Default): Max compression, skips trivial steps, strictly result-oriented.
2. symbolic: Forces variable binding and LaTeX math, minimizes text.
3. no_meta: Removes all "narrator" text, keeps only causal facts.

[NEGATIVE - For Contrast]
4. neurotic: Simulates an anxious AI planning and verifying every step.
5. micro_step: Explains simple math like a tutor for toddlers.
6. bloat: Over-defines terms and uses bureaucratic language.
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
# Defaults
# -----------------------------
REWRITE_FOLDER = "gpt_rewrites"
MODEL_NAME = "Qwen2.5-3B-Instruct"
IN_JSONL = "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_qwen_family/gsm8k_cot_zeroshot/Qwen2.5-3B-Instruct_L1_BASELINE/Qwen__Qwen2.5-3B-Instruct/samples_gsm8k_cot_zeroshot_2026-01-11T18-39-08.765074.jsonl"


def build_prompt_concise(question: str, original_resp: str) -> str:
    return f"""You will rewrite the solution to be significantly more CONCISE, reducing the token count of each step while strictly preserving the logic and meaning.

Hard constraints:
- **Minimize tokens**: Rewrite sentences to be shorter and more direct. Remove filler words, redundant adjectives, and conversational fluff.
- **Preserve Logic**: Keep the EXACT reasoning path, intermediate results, and final conclusion. Do NOT skip logical steps.
- **Preserve Math**: Keep all numbers, equations, and operations exactly as they are in the original.
- **No Style Changes**: Maintain the original tone (e.g., formal/informal), but make it denser and more efficient.
- **Structure**: You may maintain the original number of steps, but make each step shorter. 
- Preserve special markers like "<<a=b>>" if they appear.
- Output plain text only. No bullet points or added commentary.

Question:
{question}

Original solution (model output):
{original_resp}

Now output ONLY the rewritten, concise solution:
"""

def build_prompt_old(question: str, original_resp: str) -> str:
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



# =============================================================================
# PROMPT BUILDERS (POSITIVE & NEGATIVE)
# =============================================================================

def build_prompt_expert_leap(question: str, original_resp: str) -> str:
    """[POSITIVE] 极致压缩，跳过简单步骤，模拟专家直觉 (推荐用于 Layer 17)"""
    return f"""Compress the reasoning into an "Expert Summary".

1. **OMIT TRIVIAL OPS:** Do NOT show simple addition/subtraction steps (e.g., do not write "3+4=7" or "16-7=9"). Assume the reader can do mental math.
2. **FOCUS ON THE PIVOT:** Only show the setup of the final meaningful calculation.
3. **MAXIMUM COMPRESSION:** The entire solution must be fewer than 20 words if possible.
4. **DIRECT ANSWER:** Start directly with the core logic. No "First...", "Then...".

Question:
{question}

Original solution:
{original_resp}

Output the Expert Summary (plain text only):
"""

def build_prompt_symbolic(question: str, original_resp: str) -> str:
    """[POSITIVE] 强制代数符号化，最大化符号密度"""
    return f"""Rewrite the solution to maximize "Symbolic Density". Follow these rules:

1. **DEFINE & COMPUTE:** Define variables and perform the initial computation in the SAME line. Do not write "Let x be...". Just use it.
2. **NO TEXTUAL ARITHMETIC:** Never describe an operation in words (e.g., "add the blue and white fiber"). Use the equation to speak for itself.
3. **SINGLE BLOCK:** Do not use bullet points or numbered lists. The output should be 1-2 dense lines of mathematical logic.
4. **LATEX:** Use LaTeX formatting for all numbers and operations.

Question:
{question}

Original solution:
{original_resp}

Output the Symbolic Rewrite:
"""

def build_prompt_no_meta(question: str, original_resp: str) -> str:
    """[POSITIVE] 去除元认知，只保留客观因果"""
    return f"""You are a rewriting engine designed to strip away all "meta-cognitive" fluff.

1. **BANISH THE NARRATOR:** Do NOT use phrases like "First, we calculate...", "Next, we determine...", "To find the answer...", or "Step X".
2. **MERGE CAUSALITY:** Combine the cause and the effect into single sentences. Do not split "The calculation is X" and "The result is Y".
3. **REMOVE HEADERS:** Remove all bold headers.
4. **FLOW:** The output must read like a factual report, not a tutorial.

Question:
{question}

Original solution:
{original_resp}

Output the Objective Rewrite:
"""

def build_prompt_neurotic(question: str, original_resp: str) -> str:
    """[NEGATIVE] 神经质规划者，模拟过度思考 (推荐用于 Negative Sample)"""
    return f"""Rewrite the solution as an internal monologue of an AI that is over-thinking every step.

1. **DECLARE INTENT:** Start every paragraph with "I will now proceed to..." or "My next objective is..."
2. **STEP-BY-STEP LABELS:** Use nested labels like "Step 1.1: Analysis", "Step 1.2: Calculation".
3. **SELF-VERIFICATION:** After every calculation, explicitly ask yourself if it makes sense. (e.g., "Is this positive? Yes. Proceeding.")
4. **TRANSITION FLUFF:** Use long bridging sentences between steps.

Question:
{question}

Original solution:
{original_resp}

Output the Verbose Monologue:
"""

def build_prompt_micro_step(question: str, original_resp: str) -> str:
    """[NEGATIVE] 小学生保姆级，解释每一个微小步骤"""
    return f"""You are a tutor for a student who struggles with basic concepts. Rewrite the solution to be painfully explicit.

1. **VERBALIZE MATH:** Never write an equation (like 3+4=7) without first explaining it in a full sentence (e.g., "We need to combine the quantity of 3 with the quantity of 4.").
2. **RESTATE GIVENS:** Before every single step, copy-paste the specific number from the question text again.
3. **NO SHORTCUTS:** Do not combine operations. Keep them miles apart.
4. **REPETITIVE CLOSING:** End every step with a summary sentence starting with "Thus, we have determined that..."

Question:
{question}

Original solution:
{original_resp}

Output the Explicit Rewrite:
"""

def build_prompt_bloat(question: str, original_resp: str) -> str:
    """[NEGATIVE] 定义狂魔，大量废话定义"""
    return f"""Rewrite the solution to be overly formal and definition-heavy.

1. **DEFINE TERMS:** Before calculating anything, define the concept abstractly (e.g., "Revenue is defined as Price * Quantity").
2. **IDENTIFY ENTITIES:** Explicitly name the entities involved in a bureaucratic way.
3. **UNIT PEDANTRY:** After every number, write the full unit out (e.g., "16 [eggs per day]").
4. **REDUNDANT LOGIC:** State the general formula before plugging in numbers.

Question:
{question}

Original solution:
{original_resp}

Output the Formal Bloated Rewrite:
"""

# Map strategy names to functions
PROMPT_STRATEGIES = {
    "expert_leap": build_prompt_expert_leap,
    "symbolic": build_prompt_symbolic,
    "no_meta": build_prompt_no_meta,
    "neurotic": build_prompt_neurotic,
    "micro_step": build_prompt_micro_step,
    "bloat": build_prompt_bloat,
    "concise": build_prompt_concise,
    "old": build_prompt_old,
}

# =============================================================================
# MAIN SCRIPT
# =============================================================================

def get_resps_0_0(obj: Dict[str, Any]) -> Tuple[Optional[str], bool]:
    resps = obj.get("resps")
    if isinstance(resps, list) and len(resps) > 0 and isinstance(resps[0], list) and len(resps[0]) > 0:
        return (resps[0][0], True)
    return (None, False)


def set_resps_0_0(obj: Dict[str, Any], new_text: str) -> bool:
    resps = obj.get("resps")
    if isinstance(resps, list) and len(resps) > 0 and isinstance(resps[0], list) and len(resps[0]) > 0:
        obj["resps"][0][0] = new_text
        return True
    return False


def call_gpt_rewrite(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    retries: int = 4,
    sleep_base: float = 1.5,
) -> str:
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.responses.create(  # Assuming using internal or compatible OpenAI client
                model=model,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            return (resp.output_text or "").strip()
        except AttributeError:
             # Fallback for standard OpenAI Python SDK users (ChatCompletion)
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                 last_err = e
        except Exception as e:
            last_err = e
            time.sleep(sleep_base * (2 ** attempt))
    raise RuntimeError(f"OpenAI API failed after {retries} retries: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", default=IN_JSONL, help="Input JSONL file path")
    ap.add_argument("--out_json", default=None, help="Pretty JSON output path")
    ap.add_argument("--model", default="gpt-5.1", help="OpenAI model name")
    ap.add_argument("--rewrite_last_n", type=int, default=2, help="Rewrite only the last N examples")
    ap.add_argument("--overwrite_resps", action="store_true", help="Overwrite obj['resps'][0][0]")
    
    # NEW ARGUMENT: Select the prompt strategy
    ap.add_argument("--prompt_style", default="expert_leap", choices=PROMPT_STRATEGIES.keys(),
                    help="Choose the rewrite strategy. Positives: expert_leap, symbolic, no_meta. Negatives: neurotic, micro_step, bloat.")
    
    ap.add_argument("--max_output_tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY env var.", file=sys.stderr)
        sys.exit(2)

    out_json_path = os.path.join(
        REWRITE_FOLDER,
        MODEL_NAME,
        f"rewritten_{args.prompt_style}.json"
    ) if args.out_json is None else args.out_json

    os.mkdir(os.path.dirname(out_json_path)) if not os.path.exists(os.path.dirname(out_json_path)) else None

    # Select the prompt builder function based on the argument
    prompt_builder = PROMPT_STRATEGIES[args.prompt_style]
    print(f"Using Prompt Strategy: {args.prompt_style}", file=sys.stderr)

    lines = []
    with open(args.in_jsonl, "r", encoding="utf-8") as fin:
        for raw in fin:
            s = raw.strip()
            if s:
                lines.append(s)

    total = len(lines)
    start_idx = max(0, total - args.rewrite_last_n)

    client = OpenAI()
    n_attempted = 0
    n_written = 0
    n_skipped = 0
    rewritten_objs = []

    pbar = tqdm(total=total, desc="Rewriting", unit="sample", dynamic_ncols=True)

    with open(out_json_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(lines):
            obj = json.loads(line)
            obj["rewrite_index"] = i
            
            resp00, exists = get_resps_0_0(obj)
            obj["resp_before"] = resp00 if exists else ""

            # Skip if not in range
            if i < start_idx:
                obj["resp_rewrite_applied"] = False
                obj["resp_after"] = obj["resp_before"]
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                pbar.update(1)
                continue

            # Process
            n_attempted += 1
            question = (obj.get("doc") or {}).get("question", "")
            source_text = (resp00 or "").strip()

            if not exists or not question or not source_text:
                obj["resp_after"] = obj["resp_before"]
                obj["resp_rewrite_ok"] = False
                n_skipped += 1
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                pbar.update(1)
                continue

            # CALL API with selected prompt style
            prompt = prompt_builder(question, source_text)
            
            try:
                rewritten = call_gpt_rewrite(
                    client=client,
                    model=args.model,
                    prompt=prompt,
                    max_tokens=args.max_output_tokens,
                    temperature=args.temperature,
                )
                obj["resp_after"] = rewritten
                obj["resp_rewrite_ok"] = True
                obj["resp_rewrite_style"] = args.prompt_style  # Record which style was used
                n_written += 1

                if args.overwrite_resps:
                    set_resps_0_0(obj, rewritten)

            except Exception as e:
                print(f"\nError processing index {i}: {e}", file=sys.stderr)
                obj["resp_rewrite_ok"] = False
                obj["resp_after"] = obj["resp_before"]

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            rewritten_objs.append(obj)
            
            if args.sleep > 0:
                time.sleep(args.sleep)
            pbar.update(1)

    pbar.close()

    with open(out_json_path, "w", encoding="utf-8") as fjson:
        json.dump(rewritten_objs, fjson, ensure_ascii=False, indent=2)

    print(f"Done. Rewrote {n_written} samples using style '{args.prompt_style}'.", file=sys.stderr)

if __name__ == "__main__":
    main()