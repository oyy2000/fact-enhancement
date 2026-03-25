#!/usr/bin/env python3
"""
Dense rewriting using an open-source model instead of GPT-5.1.
Uses the same rewriting prompts but with a local HuggingFace model.

This addresses reviewer concerns about dependency on closed-source models
and potential hidden knowledge transfer from GPT-5.1.
"""

import argparse
import json
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


QWEN_3B_GPT_REWRITE = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/gpt_rewrites_unified_new/Qwen_Qwen2.5-3B-Instruct/rewritten_old.json"

def build_prompt_old(question: str, original_resp: str) -> str:
    return f"""You will lightly rewrite the solution by CONSERVATIVELY merging steps, while keeping the SAME style and meaning.

Hard constraints:
- Keep the SAME meaning and do NOT change the final conclusion/answer implied by the solution.
- Do NOT invent new reasoning. Only compress/merge/rephrase existing steps.
- Keep the style and tone the SAME as the original (do not change formality, phrasing habits, or formatting conventions).
- Only merge steps when it is NECESSARY and safe (e.g., two adjacent lines that are clearly redundant or tightly coupled).
  Do NOT aggressively minimize the number of steps. If merging would change the "feel" or clarity, keep the original steps.
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

def build_prompt_reflow(question: str, original_resp: str) -> str:
    return f"""You will lightly rewrite the solution by CONSERVATIVELY merging steps by removing single newline characters (`\\n`) as much as possible, while keeping the SAME style and meaning.

What to change:
- Detect single newline characters (`\\n`) that interrupt a continuous sentence, mathematical reasoning, or logical flow.
- Replace such single newlines with a space, OR minimally rephrase to smoothly merge the lines if required for grammatical correctness.

What to preserve:
- Keep double newlines (`\\n\\n`) or any formatting that clearly separates paragraphs, steps, or list items.
- Preserve the original reasoning structure, order of steps, and writing style.

Hard constraints:
- Do NOT change any numbers, equations, calculations, or final answers.
- Do NOT add new reasoning or remove existing reasoning.
- Ensure the merged text is grammatically correct and semantically identical to the original.

Output format:
- Output plain text only.
- Output ONLY the rewritten solution, with no explanations, comments, or extra text.

Question:
{question}

Original solution:
{original_resp}

Rewritten solution:
"""


PROMPT_STYLES = {
    "old": build_prompt_old,
    "reflow": build_prompt_reflow,
}


def load_samples(path, max_samples=None):
    """Load samples from GPT rewrite JSON (which already has resp_before)."""
    with open(path, "r") as f:
        data = json.load(f)
    
    result = []
    for obj in data:
        question = obj.get("doc", {}).get("question", "")
        resp_before = obj.get("resp_before", "")
        if question.strip() and resp_before.strip():
            result.append({
                "doc_id": obj.get("doc_id"),
                "question": question,
                "resp_before": resp_before,
                "doc": obj.get("doc", {}),
            })
        if max_samples and len(result) >= max_samples:
            break
    return result


def rewrite_with_local_model(model, tokenizer, samples, prompt_style, max_new_tokens=2048):
    """Rewrite samples using a local HuggingFace model."""
    prompt_fn = PROMPT_STYLES[prompt_style]
    results = []

    for sample in tqdm(samples, desc=f"Rewriting ({prompt_style})"):
        prompt = prompt_fn(sample["question"], sample["resp_before"])

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        rewritten = tokenizer.decode(generated, skip_special_tokens=True).strip()

        results.append({
            "doc_id": sample["doc_id"],
            "question": sample["question"],
            "resp_before": sample["resp_before"],
            "resp_after": rewritten,
            "resp_rewrite_ok": True,
            "resp_rewrite_style": prompt_style,
            "doc": sample["doc"],
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewriter_model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Open-source model to use for rewriting")
    parser.add_argument("--target_model", default="Qwen/Qwen2.5-3B-Instruct",
                        help="Target model whose outputs we are rewriting")
    parser.add_argument("--samples_path", default=QWEN_3B_GPT_REWRITE,
                        help="Path to GPT rewrite JSON (has resp_before fields)")
    parser.add_argument("--prompt_style", default="old", choices=PROMPT_STYLES.keys())
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Max correct samples to rewrite")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    rewriter_safe = args.rewriter_model.replace("/", "_")
    target_safe = args.target_model.replace("/", "_")

    if args.output_dir is None:
        args.output_dir = os.path.join(
            "opensource_rewrites", target_safe
        )
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading rewriter model: {args.rewriter_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.rewriter_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.rewriter_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"Loading samples from: {args.samples_path}")
    samples = load_samples(args.samples_path, max_samples=args.max_samples)
    print(f"Loaded {len(samples)} correct samples")

    results = rewrite_with_local_model(model, tokenizer, samples, args.prompt_style)

    out_path = os.path.join(args.output_dir, f"rewritten_{args.prompt_style}_by_{rewriter_safe}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} rewritten samples to {out_path}")


if __name__ == "__main__":
    main()
