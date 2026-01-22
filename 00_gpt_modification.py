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
REWRITE_FOLDER = "gpt_rewrites_unified"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct".replace("/", "_")
IN_JSONL = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/no_vector/gsm8k_cot_zeroshot_unified/Qwen2.5-3B-Instruct_no_vector/Qwen__Qwen2.5-3B-Instruct/samples_gsm8k_cot_zeroshot_unified_2026-01-21T11-33-55.190404.jsonl"


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


def build_prompt_explicit_reference(question: str, original_resp: str) -> str:
    """
    [NEGATIVE 1] Force explicit back-referencing.
    Constraint: Add sentences that explicitly state "Using the result from Step X..."
    """
    return f"""Rewrite the solution to be overly explicit about data flow.

    **Goal:** Simulate a model with low working memory that must repeat previous results to keep track of them.

    **Rules:**
    1. **INSERT BRIDGING SENTENCES:** At the start of every new calculation step, insert a full sentence explicitly stating which numbers from the *previous* step are being used.
       - *Example:* "Using the total of 7 eggs calculated in the previous step..."
    2. **REPEAT VALUES:** Do not just say "that number". Repeat the actual number (e.g., "Using the value of 18...").
    3. **PRESERVE STRUCTURE:** Keep the original step headers and general logic intact, just add these "glue" sentences.

    ### Example:
    **Input:** "Step 2: 16 - 7 = 9."
    **Target:** "Step 2: We now take the total of 16 and the used amount of 7. Using the result of 7 calculated in Step 1, we subtract it from 16: 16 - 7 = 9."

    **Task:**
    **Question:** {question}
    **Original Response:** {original_resp}
    
    **Output:**
    """


def build_prompt_implicit_context(question: str, original_resp: str) -> str:
    """
    [POSITIVE 1] Remove explicit references, rely on implicit context.
    Constraint: Logic must flow without pointing backwards.
    """
    return f"""Rewrite the solution to remove all explicit "back-references" while keeping the logic strictly correct.

    **Goal:** Simulate a high-context model that remembers the state without needing to repeat it.

    **Rules:**
    1. **DELETE REFERENCES:** Remove all phrases like "Using the result from step 1", "As calculated above", "Recall that x is 5".
    2. **SEAMLESS LOGIC:** Ensure the sentence still makes sense grammatically.
       - *Bad:* "Subtract it from 16." (What is 'it'?)
       - *Good:* "Subtracting this usage from the daily total of 16 yields 9."
    3. **NO STRUCTURAL CHANGE:** Do not change the calculation steps themselves, only the text *describing* the data flow.

    **Task:**
    **Question:** {question}
    **Original Response:** {original_resp}
    
    **Output:**
    """


def build_prompt_mimic_14b(question: str, original_resp: str) -> str:
    """
    [POSITIVE 2] Mimic 14B Model Style (High Intelligence).
    Constraint: Narrative flow, embedded verbs, no scaffolding labels.
    """
    return f"""Rewrite the solution to mimic the style of a 14B parameter model (High Density, Narrative Flow).

    **Strict Constraints (Based on 14B Statistics):**
    1. **FORBIDDEN WORDS:** You must NOT use the words "Step", "Label", or start sentences with distinct imperative verbs like "Calculate" or "Determine".
    2. **NARRATIVE FLOW:** You MUST use transition words: "First,", "Next,", "Then,", "Finally,".
    3. **EMBEDDED VERBS:** Do not issue commands. Embed the calculation in the flow.
       - *Bad:* "Calculate the cost. 5 * 2 = 10."
       - *Good:* "Calculating the cost yields $5 \\times 2 = 10$." or "We first calculate..."
    4. **DIRECT COMPUTATION:** Avoid "Let variable be...". Use numbers and concepts directly to reduce token count.

    ### Example (Target Style):
    "First, calculating the total usage (3 breakfast + 4 muffins) gives 7 eggs. Next, subtracting this from the daily 16 leaves 9 eggs. Finally, selling these at $2 each results in $18."

    **Task:**
    **Question:** {question}
    **Original Response:** {original_resp}
    
    **Output:**
    """


def build_prompt_mimic_3b_7b(question: str, original_resp: str) -> str:
    """
    [NEGATIVE 2] Mimic 3B/7B Model Style (Pedantic Scaffolding).
    Constraint: Rigid steps, imperative commands, explicit variable definitions.
    """
    return f"""Rewrite the solution to mimic the style of a 3B/7B parameter model (Rigid Scaffolding, Low Confidence).

    **Strict Constraints (Based on 3B/7B Statistics):**
    1. **MANDATORY LABELS:** Every logical unit MUST start with "**Step N:**".
    2. **IMPERATIVE OPENERS:** The text following the label must start with a capitalized imperative verb (e.g., "Calculate the...", "Determine the...").
    3. **VARIABLE DECLARATION:** You must explicit define variables before using them (e.g., "Let $x$ be...").
    4. **VERBOSE DEFINITIONS:** Over-explain simple terms to fill space.

    ### Example (Target Style):
    "**Step 1:** Calculate the total eggs used.
    Let $B$ be eggs for breakfast (3) and $M$ be eggs for muffins (4). We calculate the sum: $3 + 4 = 7$.
    
    **Step 2:** Determine the remainder.
    Let $T$ be the total eggs (16). We subtract the result from Step 1: $16 - 7 = 9$."

    **Task:**
    **Question:** {question}
    **Original Response:** {original_resp}
    
    **Output:**
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
        f"{MODEL_NAME}",
        f"rewritten_{args.prompt_style}.json"
    ) if args.out_json is None else args.out_json

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)

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