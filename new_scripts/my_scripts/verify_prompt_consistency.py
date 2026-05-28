#!/usr/bin/env python3
"""
Verify that extraction prompts and eval (lm_eval) prompts are identical
for both Qwen and Llama models.
"""
import sys
sys.path.insert(0, "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement")
sys.path.insert(0, "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/new_scripts/my_scripts")
from utils import qwen_chat_prompt
from e8_extract_vectors import make_prompt, DOC_TO_TEXT_TEMPLATE
from transformers import AutoTokenizer

TEST_QUESTION = "Janet's ducks lay 16 eggs per day."


def build_eval_prompt(tokenizer, question):
    """Replicate what lm_eval does: doc_to_text + apply_chat_template."""
    user_content = DOC_TO_TEXT_TEMPLATE.format(question=question)
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def compare(label, a, b):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"\n--- EXTRACTION prompt ({len(a)} chars) ---")
    print(repr(a))
    print(f"\n--- EVAL prompt ({len(b)} chars) ---")
    print(repr(b))

    if a == b:
        print(f"\n>>> MATCH: extraction == eval")
    else:
        print(f"\n>>> MISMATCH! Showing diff:")
        min_len = min(len(a), len(b))
        for i in range(min_len):
            if a[i] != b[i]:
                ctx = 30
                print(f"    First diff at index {i}:")
                print(f"      extraction: ...{repr(a[max(0,i-ctx):i+ctx])}...")
                print(f"      eval:       ...{repr(b[max(0,i-ctx):i+ctx])}...")
                break
        if len(a) != len(b):
            print(f"    Length diff: extraction={len(a)}, eval={len(b)}")
    return a == b


def test_model(model_name):
    print(f"\n{'#'*70}")
    print(f"# {model_name}")
    print(f"{'#'*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Extraction: uses make_prompt(tokenizer, question) after fix
    extract_prompt = make_prompt(tokenizer, TEST_QUESTION)

    # Eval: lm_eval uses doc_to_text + apply_chat_template
    eval_prompt = build_eval_prompt(tokenizer, TEST_QUESTION)

    ok = compare(f"{model_name}: make_prompt vs eval", extract_prompt, eval_prompt)

    # Also test qwen_chat_prompt for Qwen models (used by other scripts)
    if "qwen" in model_name.lower():
        qwen_prompt = qwen_chat_prompt(TEST_QUESTION)
        ok2 = compare(f"{model_name}: qwen_chat_prompt vs eval", qwen_prompt, eval_prompt)
        return ok and ok2

    return ok


if __name__ == "__main__":
    models = [
        "Qwen/Qwen2.5-3B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
    ]

    results = {}
    for m in models:
        results[m] = test_model(m)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    all_ok = True
    for m, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {m}: {status}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\nALL PROMPTS CONSISTENT.")
    else:
        print("\nMISMATCH DETECTED - FIX NEEDED.")
