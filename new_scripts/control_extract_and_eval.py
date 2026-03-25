"""
Extract steering vectors from control experiment data and evaluate on GSM8K.
This is the full pipeline for the random compression control experiment.

Usage:
  python control_extract_and_eval.py --model Qwen/Qwen2.5-3B-Instruct --layer 6 --lam 4.0
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

LLAMA_CHAT_TEMPLATE = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nSolve the following math problem. Present the final answer in the format: Final Answer: \\boxed{{your_answer}}.\nProlbem: {question}\nAnswer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def make_prompt(model_name, question):
    if "llama" in model_name.lower():
        return LLAMA_CHAT_TEMPLATE.format(question=question)
    return qwen_chat_prompt(question)


def extract_control_vector(model_name, control_data_path, output_dir,
                           num_examples=50):
    """Extract steering vector from control experiment data."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    ).eval()

    data = json.load(open(control_data_path))
    
    selected = [ex for ex in data if ex.get("exact_match", 0) == 1.0][-num_examples:]
    print(f"Using {len(selected)} samples for vector extraction")

    if "llama" in model_name.lower():
        num_layers = model.config.num_hidden_layers
    else:
        num_layers = model.config.num_hidden_layers
    layer_list = list(range(num_layers))

    training_samples = []
    for ex in selected:
        question = ex["doc"]["question"]
        prompt = make_prompt(model_name, question)
        training_samples.append((
            prompt + ex["resp_after"],
            prompt + ex["resp_before"]
        ))

    print(f"Training steering vector for {len(layer_list)} layers...")
    steering_vector = train_steering_vector(
        model=model, tokenizer=tokenizer,
        training_samples=training_samples,
        layers=layer_list, layer_type="decoder_block",
        move_to_cpu=True, read_token_index=-1,
        show_progress=True, batch_size=1
    )

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "steering_vector.pt")
    torch.save(steering_vector, save_path)
    print(f"Saved steering vector to {save_path}")

    del model
    torch.cuda.empty_cache()
    return save_path


def evaluate_gsm8k(model_name, vec_path, layer, lam, output_dir, gpu=0):
    """Evaluate on GSM8K using lm_eval with the control steering vector."""
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

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(result.stdout[-500:] if result.stdout else "")
    if result.returncode != 0:
        print(f"ERROR: {result.stderr[-500:]}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model name")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--lam", type=float, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip_extract", action="store_true")
    parser.add_argument("--compression", default="random_merge")
    args = parser.parse_args()

    model_folder = args.model.replace("/", "_")
    control_data = os.path.join(BASE, "control_experiments", model_folder,
                                 f"rewritten_control_{args.compression}.json")
    vec_dir = os.path.join(BASE, "control_experiments", model_folder,
                            f"vectors_{args.compression}")
    vec_path = os.path.join(vec_dir, "steering_vector.pt")

    if not args.skip_extract:
        vec_path = extract_control_vector(
            args.model, control_data, vec_dir)

    eval_dir = os.path.join(BASE, "control_experiments", model_folder,
                             f"eval_{args.compression}_L{args.layer}_lam{args.lam}")
    evaluate_gsm8k(args.model, vec_path, args.layer, args.lam,
                    eval_dir, args.gpu)


if __name__ == "__main__":
    main()
