"""
Calibration Set Size Ablation
Addresses Reviewer jRak's question: "What is the effect of calibration set size?"

Extracts steering vectors using N = {1, 5, 10, 25, 50} pairs
and evaluates each on GSM8K.
"""
import torch
import json
import os
import argparse
import subprocess
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from steering_vectors import train_steering_vector
from utils import qwen_chat_prompt

BASE = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement"


def make_prompt(model_name, question):
    if "llama" in model_name.lower():
        return LLAMA_CHAT_TEMPLATE.format(question=question)
    return qwen_chat_prompt(question)


def get_exact_match(ex):
    if "exact_match" in ex:
        try:
            return float(ex["exact_match"])
        except (TypeError, ValueError):
            pass
    return None


def extract_vector_for_n(model, tokenizer, model_name, samples, n, layer_list, output_dir):
    """Extract a steering vector using exactly n samples."""
    selected = samples[-n:] if len(samples) >= n else samples
    
    training_pairs = []
    for ex in selected:
        question = ex["doc"]["question"]
        prompt = make_prompt(model_name, question)
        training_pairs.append((
            prompt + ex["resp_after"],
            prompt + ex["resp_before"]
        ))

    sv = train_steering_vector(
        model=model, tokenizer=tokenizer,
        training_samples=training_pairs,
        layers=layer_list, layer_type="decoder_block",
        move_to_cpu=True, read_token_index=-1,
        show_progress=True, batch_size=1
    )

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "steering_vector.pt")
    torch.save(sv, save_path)
    return save_path


def evaluate_gsm8k(model_name, vec_path, layer, lam, output_dir, gpu=0):
    """Evaluate on GSM8K."""
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "lm_eval",
        "--model", "steer_hf",
        "--model_args", f"pretrained={model_name},dtype=float16,"
                        f"steer_layer={layer},steer_lambda={lam},"
                        f"steer_vec_path={vec_path},"
                        "trust_remote_code=True",
        "--tasks", "gsm8k_cot_zeroshot_unified",
        "--batch_size", "64",
        "--num_fewshot", "0",
        "--output_path", output_dir,
        "--log_samples",
        "--trust_remote_code",
        "--gen_kwargs", "do_sample=False,temperature=0,max_gen_toks=2048",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    print(f"  Evaluating N={os.path.basename(output_dir)}...")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    for f in os.listdir(output_dir):
        if f.startswith("results_") and f.endswith(".json"):
            with open(os.path.join(output_dir, f)) as fp:
                data = json.load(fp)
            acc = data.get("results", {}).get("gsm8k_cot_zeroshot_unified", {}).get(
                "exact_match,flexible-extract", "N/A")
            print(f"    Accuracy: {acc}")
            return acc
    
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr[-300:]}")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--mode", default="GPT_REWRITE",
                        choices=["GPT_REWRITE", "LARGE_MODEL"])
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--lam", type=float, default=4.0)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[1, 5, 10, 25, 50])
    args = parser.parse_args()

    model_folder = args.model.replace("/", "_")
    
    if args.mode == "GPT_REWRITE":
        sample_path = os.path.join(BASE, "gpt_rewrites_unified_new",
                                    model_folder, "rewritten_old.json")
    else:
        raise NotImplementedError("LARGE_MODEL mode not yet implemented for ablation")

    data = json.load(open(sample_path))
    em1 = [ex for ex in data if get_exact_match(ex) == 1.0]
    print(f"Total EM=1 samples available: {len(em1)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    ).eval()

    num_layers = model.config.num_hidden_layers
    layer_list = list(range(num_layers))

    ablation_dir = os.path.join(BASE, "calibration_ablation", model_folder, args.mode)
    results_summary = []

    for n in args.sizes:
        print(f"\n--- N = {n} ---")
        vec_dir = os.path.join(ablation_dir, f"vectors_N{n}")
        vec_path = extract_vector_for_n(
            model, tokenizer, args.model, em1, n, layer_list, vec_dir
        )

    del model
    torch.cuda.empty_cache()

    for n in args.sizes:
        vec_dir = os.path.join(ablation_dir, f"vectors_N{n}")
        vec_path = os.path.join(vec_dir, "steering_vector.pt")
        eval_dir = os.path.join(ablation_dir, f"eval_N{n}_L{args.layer}_lam{args.lam}")
        
        acc = evaluate_gsm8k(args.model, vec_path, args.layer, args.lam,
                              eval_dir, args.gpu)
        results_summary.append({"N": n, "accuracy": acc})

    print("\n" + "=" * 50)
    print("CALIBRATION SIZE ABLATION SUMMARY")
    print("=" * 50)
    for r in results_summary:
        acc_str = f"{r['accuracy']*100:.1f}%" if isinstance(r['accuracy'], float) else "N/A"
        print(f"  N = {r['N']:3d}: {acc_str}")

    summary_path = os.path.join(ablation_dir, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
