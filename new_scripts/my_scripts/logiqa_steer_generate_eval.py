#!/usr/bin/env python3
"""
Generate CoT for LogiQA with steering, extract answers, compute accuracy.
Uses HF model + steering hook (same as steer_hf in lm_eval) + vLLM-style batched generation.
"""
import argparse, json, re, sys, os
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))
from utils import logiqa_format_problem, logiqa_qwen_chat_prompt


def parse_final_answer(text: str):
    if not text:
        return None
    m = re.search(r"Final Answer:\s*([A-Da-d])\b", text, re.I)
    if m:
        return m.group(1).upper()
    for pat in [
        r"(?:correct\s+)?(?:answer|option)\s+is\s*:?\s*\*?\s*([A-Da-d])\b",
        r"\b(?:choose|select|pick)\s+(?:option\s+)?([A-Da-d])\b",
        r"\b(?:Therefore|Thus|Hence),?\s+(?:the\s+)?(?:answer|correct\s+option)\s+is\s*:?\s*([A-Da-d])\b",
    ]:
        m2 = re.search(pat, text, re.I)
        if m2:
            return m2.group(1).upper()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for ln in reversed(lines[-8:]):
        m3 = re.match(r"^([A-Da-d])\s*[\.\)]?$", ln)
        if m3:
            return m3.group(1).upper()
    m4 = re.findall(r"\b([A-D])\b", text[-200:])
    if m4:
        return m4[-1].upper()
    return None


def label_to_letter(label):
    if isinstance(label, int):
        return chr(65 + label)
    s = str(label).strip().upper()
    if len(s) == 1 and s in "ABCD":
        return s
    if s in ['0','1','2','3']:
        return chr(65 + int(s))
    return "?"


class SteeringHook:
    """Context manager that adds a steering vector to a specific layer."""
    def __init__(self, model, vec_path, layer, lam):
        self.model = model
        self.layer = layer
        self.lam = lam
        self.handle = None
        if vec_path and lam != 0 and layer is not None:
            raw = torch.load(vec_path, map_location="cpu")
            if isinstance(raw, dict):
                self.vec = raw.get(f"layer_{layer}", raw.get(layer, None))
                if self.vec is None:
                    # try first key
                    self.vec = list(raw.values())[0]
            else:
                self.vec = raw
            self.vec = self.vec.to(dtype=torch.float16)
        else:
            self.vec = None

    def __enter__(self):
        if self.vec is None or self.lam == 0:
            return self
        device = next(self.model.parameters()).device
        vec = self.vec.to(device)

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
                h = h + self.lam * vec.unsqueeze(0).unsqueeze(0)
                return (h,) + output[1:]
            else:
                return output + self.lam * vec.unsqueeze(0).unsqueeze(0)

        # Find the target layer
        layers = None
        for attr in ['model.layers', 'transformer.h', 'gpt_neox.layers']:
            obj = self.model
            for part in attr.split('.'):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                layers = obj
                break
        
        if layers is not None and self.layer < len(layers):
            self.handle = layers[self.layer].register_forward_hook(hook_fn)
        return self

    def __exit__(self, *args):
        if self.handle:
            self.handle.remove()
            self.handle = None


def generate_batch(model, tokenizer, prompts, max_new_tokens=768):
    """Generate responses for a batch of prompts."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=3072)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    responses = []
    for i, out in enumerate(outputs):
        input_len = inputs['input_ids'][i].shape[0]
        gen_tokens = out[input_len:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        responses.append(text)
    return responses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--vec_path", default="", help="Path to steering_vector.pt")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--out_jsonl", default="")
    args = ap.parse_args()

    print(f"Loading dataset...", flush=True)
    ds = load_dataset("EleutherAI/logiqa", "logiqa", split="test", trust_remote_code=True)
    ds = ds.select(range(min(args.limit, len(ds))))

    print(f"Loading model {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    # Prepare prompts
    prompts = []
    golds = []
    for i in range(len(ds)):
        row = ds[i]
        prompt = logiqa_qwen_chat_prompt(row["context"], row["question"], list(row["options"]))
        prompts.append(prompt)
        golds.append(label_to_letter(row["label"]))

    # Generate with steering
    tag = f"L{args.layer}_lam{args.lam}" if args.vec_path else "baseline"
    print(f"Generating {len(prompts)} samples [{tag}]...", flush=True)
    
    hook = SteeringHook(model, args.vec_path, args.layer, args.lam)
    
    all_responses = []
    with hook:
        for start in range(0, len(prompts), args.batch_size):
            end = min(start + args.batch_size, len(prompts))
            batch = prompts[start:end]
            resps = generate_batch(model, tokenizer, batch, args.max_new_tokens)
            all_responses.extend(resps)
            if (start // args.batch_size) % 5 == 0:
                print(f"  {start+len(batch)}/{len(prompts)}", flush=True)

    # Evaluate
    correct = 0
    results = []
    for i, (resp, gold) in enumerate(zip(all_responses, golds)):
        pred = parse_final_answer(resp)
        em = 1.0 if pred and pred == gold else 0.0
        if em:
            correct += 1
        results.append({
            "doc_id": i,
            "gold": gold,
            "pred": pred,
            "exact_match": em,
            "response": resp,
        })

    acc = correct / len(results)
    print(f"\n{tag}: {correct}/{len(results)} = {acc*100:.2f}%", flush=True)

    if args.out_jsonl:
        Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_jsonl, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved {args.out_jsonl}")

    # Print summary
    print(json.dumps({
        "model": args.model,
        "vec_path": args.vec_path,
        "layer": args.layer,
        "lam": args.lam,
        "n": len(results),
        "correct": correct,
        "acc": round(acc, 4),
    }))


if __name__ == "__main__":
    main()
