#!/usr/bin/env python3
"""
Unified multi-dataset sampling for Figure 1 validation.
Supports: GSM8K, MATH-500, AIME, AMC, Olympiad.
Uses vLLM for fast multi-GPU inference.

Usage:
    CUDA_VISIBLE_DEVICES=0 python figure1_multi_dataset_vllm.py \
        --model Qwen/Qwen2.5-3B-Instruct --dataset math500
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

BASE = Path("/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement")
OUT_ROOT = BASE / "new_exps" / "figure1_multi_dataset"
DATA_ROOT = BASE / "exps" / "math_eval_data"

PROMPT_TEMPLATE = (
    "Solve the following math problem. Present the final answer in the "
    "format: Final Answer: \\boxed{{your_answer}}.\n"
    "Problem: {question}\nAnswer:"
)

DATASET_CONFIGS = {
    "gsm8k": {
        "loader": "huggingface",
        "hf_path": "gsm8k",
        "hf_name": "main",
        "split": "test",
        "question_field": "question",
        "answer_field": "answer",
        "answer_extractor": "gsm8k",
    },
    "math500": {
        "loader": "jsonl",
        "path": DATA_ROOT / "MATH-500" / "test.jsonl",
        "question_field": "problem",
        "answer_field": "answer",
        "answer_extractor": "math",
    },
    "aime": {
        "loader": "jsonl",
        "path": DATA_ROOT / "aime24" / "test.jsonl",
        "question_field": "problem",
        "answer_field": "answer",
        "answer_extractor": "math",
    },
    "amc": {
        "loader": "jsonl",
        "path": DATA_ROOT / "amc23" / "test.jsonl",
        "question_field": "problem",
        "answer_field": "answer",
        "answer_extractor": "math",
    },
    "olympiad": {
        "loader": "jsonl",
        "path": DATA_ROOT / "olympiadbench" / "test.jsonl",
        "question_field": "question",
        "answer_field": "final_answer",
        "answer_extractor": "math",
    },
}


# ── Math answer comparison (from lm-evaluation-harness) ──────────────────────

def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except (AssertionError, AssertionError):
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    new_str += "{" + a + "}{" + b + "}" + substr[2:]
                else:
                    new_str += "{" + a + "}" + b + substr[2:]
    return new_str


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a, b = int(a), int(b)
        assert string == f"{a}/{b}"
        return f"\\frac{{{a}}}{{{b}}}"
    except (ValueError, AssertionError):
        return string


def remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if len(split) > 0 and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = fix_a_slash_b(string)
    return string


def is_equiv(str1, str2):
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        str1 = str1.lstrip("0")
        str2 = str2.lstrip("0")
        if "." in str1 and str1.split(".")[1].rstrip("0") == "":
            str1 = str1.split(".")[0]
        if "." in str2 and str2.split(".")[1].rstrip("0") == "":
            str2 = str2.split(".")[0]
        return strip_string(str1) == strip_string(str2)
    except Exception:
        return str1 == str2


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(s):
    if "\\fbox{" in s:
        left = "\\fbox{"
        if s[: len(left)] == left and s[-1] == "}":
            return s[len(left) : -1]
    if "\\boxed " in s:
        left = "\\boxed "
        if s[: len(left)] == left:
            return s[len(left) :]
    left = "\\boxed{"
    return s[len(left) : -1]


# ── Answer extraction ────────────────────────────────────────────────────────

def extract_boxed_answer(text: str) -> str | None:
    boxed = last_boxed_only_string(text)
    if boxed is not None:
        return remove_boxed(boxed)
    match = re.search(r"Final Answer:\s*(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_gsm8k_gold(answer_str: str) -> str:
    match = re.search(r"####\s*(.+)", answer_str)
    if match:
        return match.group(1).strip().replace(",", "")
    return answer_str.strip()


def check_correct(pred_str, gold_str, method="math"):
    if pred_str is None:
        return False
    if method == "gsm8k":
        def norm(s):
            s = s.strip().rstrip(".").replace(",", "").replace("$", "").replace("%", "").replace(" ", "")
            try:
                return str(float(s))
            except ValueError:
                return s.lower()
        return norm(pred_str) == norm(gold_str)
    else:
        return is_equiv(pred_str, str(gold_str))


# ── Step & density computation ───────────────────────────────────────────────

def count_steps_and_density(text: str, tokenizer):
    steps = [s.strip() for s in text.split("\n\n") if s.strip()]
    if not steps:
        return 1, 0, 0, 0
    n_steps = len(steps)
    step_token_lens = [len(tokenizer.encode(step, add_special_tokens=False)) for step in steps]
    total_tokens = sum(step_token_lens)
    avg_tokens_per_step = float(np.mean(step_token_lens))
    density = total_tokens / n_steps
    return n_steps, total_tokens, avg_tokens_per_step, density


# ── Data loading ─────────────────────────────────────────────────────────────

def load_dataset_items(dataset_name):
    cfg = DATASET_CONFIGS[dataset_name]
    questions, gold_answers = [], []

    if cfg["loader"] == "huggingface":
        from datasets import load_dataset
        ds = load_dataset(cfg["hf_path"], cfg["hf_name"], split=cfg["split"])
        for ex in ds:
            questions.append(ex[cfg["question_field"]])
            gold_answers.append(extract_gsm8k_gold(ex[cfg["answer_field"]]))
    else:
        with open(cfg["path"], encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                questions.append(doc[cfg["question_field"]])
                raw_ans = doc[cfg["answer_field"]]
                if isinstance(raw_ans, list):
                    raw_ans = raw_ans[0]
                gold_answers.append(str(raw_ans))

    return questions, gold_answers, cfg["answer_extractor"]


# ── Prompt building ──────────────────────────────────────────────────────────

def get_system_prompt(model_name: str) -> str:
    model_lower = model_name.lower()
    if "qwen" in model_lower:
        return "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    return "You are a helpful assistant."


def build_chat_prompt(question: str, tokenizer, model_name: str = "") -> str:
    user_msg = PROMPT_TEMPLATE.format(question=question)
    messages = [
        {"role": "system", "content": get_system_prompt(model_name)},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--swap_space", type=int, default=4,
                        help="CPU swap space per GPU in GB (default: 4)")
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    model_sanitized = args.model.replace("/", "_")
    out_dir = OUT_ROOT / args.dataset / model_sanitized
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "samples.jsonl"

    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Tensor parallel: {args.tensor_parallel_size} GPUs")
    print(f"Samples per question: {args.num_samples}")
    print(f"Output: {out_path}")

    questions, gold_answers, answer_method = load_dataset_items(args.dataset)
    if args.limit:
        questions = questions[:args.limit]
        gold_answers = gold_answers[:args.limit]
    print(f"Loaded {len(questions)} questions")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("Loading vLLM engine...")
    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="float16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=args.swap_space,
    )
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|eot_id|>", "Problem:"],
    )

    prompts = [build_chat_prompt(q, tokenizer, args.model) for q in questions]

    print(f"Generating {len(prompts)} x {args.num_samples} = {len(prompts) * args.num_samples} responses...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    print(f"Generation done in {elapsed:.0f}s ({len(prompts) * args.num_samples / elapsed:.1f} samples/s)")

    with open(out_path, "w", encoding="utf-8") as fout:
        for doc_id, (output, question, gold) in enumerate(
            zip(outputs, questions, gold_answers)
        ):
            sample_records = []
            for s_idx, completion in enumerate(output.outputs):
                resp_text = completion.text
                pred = extract_boxed_answer(resp_text)
                correct = check_correct(pred, gold, method=answer_method)
                n_steps, total_toks, avg_tps, density = count_steps_and_density(
                    resp_text, tokenizer
                )
                sample_records.append({
                    "sample_idx": s_idx,
                    "response": resp_text,
                    "predicted_answer": pred,
                    "gold_answer": gold,
                    "correct": correct,
                    "n_steps": n_steps,
                    "total_tokens": total_toks,
                    "avg_tokens_per_step": round(avg_tps, 2),
                    "density_rho": round(density, 2),
                })

            record = {
                "doc_id": doc_id,
                "question": question,
                "gold_answer": gold,
                "dataset": args.dataset,
                "model": args.model,
                "num_samples": args.num_samples,
                "temperature": args.temperature,
                "samples": sample_records,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    total_samples = 0
    total_correct = 0
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            for s in obj["samples"]:
                total_samples += 1
                if s["correct"]:
                    total_correct += 1

    print(f"\nResults: {total_samples} samples, {total_correct} correct, "
          f"pass@1 ≈ {total_correct / total_samples:.3f}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
