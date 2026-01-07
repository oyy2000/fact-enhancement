import json
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import os
import random

# ============================================================
# 🔒 FIXED RANDOM SEED
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONHASHSEED"] = str(SEED)

# ============================================================
# PRM MODEL (ONLY FOR SCORING)
# ============================================================
PRM_MODEL = "Qwen/Qwen2.5-Math-PRM-7B"

prm_tokenizer = AutoTokenizer.from_pretrained(
    PRM_MODEL, trust_remote_code=True
)
prm_model = AutoModel.from_pretrained(
    PRM_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
).eval()

STEP_TOKEN = prm_tokenizer.encode("<extra_0>")[0]

# ============================================================
# PRM SCORING
# ============================================================
def prm_step_scores(logits, input_ids):
    probs = F.softmax(logits, dim=-1)
    idx = (input_ids == STEP_TOKEN).nonzero(as_tuple=True)[1]
    return [
        probs[0, idx[i]:idx[i + 1], 1].mean().item()
        for i in range(len(idx) - 1)
    ]


def eval_cot_prm(query, steps):
    """
    Only used to score steps, NOT for token counting
    """
    text = "<extra_0>".join(steps) + "<extra_0>"
    msgs = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": text}
    ]
    conv = prm_tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False
    )
    ids = prm_tokenizer.encode(conv, return_tensors="pt").to(prm_model.device)

    with torch.no_grad():
        out = prm_model(input_ids=ids, use_cache=False)

    return prm_step_scores(out[0], ids)


# ============================================================
# MAIN DATASET RUNNER (RECORD ONLY)
# ============================================================
def run_dataset(jsonl_path, gen_model_name, label="exact_match", eval_start=100):
    """
    jsonl_path: path to lm-eval generated jsonl
    gen_model_name: HF repo id of the model that generated the CoT
    eval_start: start evaluating from this sample index (default=100)
    """

    # decoder tokenizer (for token counting)
    gen_tokenizer = AutoTokenizer.from_pretrained(
        gen_model_name,
        trust_remote_code=True,
        use_fast=True
    )

    data = [json.loads(l) for l in open(jsonl_path)]

    Y = []
    STEP_SCORES_ALL = []
    STEP_TEXTS_ALL = []
    STEP_TOKEN_LEN = []

    # ============================================================
    # MAIN LOOP (START FROM eval_start)
    # ============================================================
    for idx, d in enumerate(data):

        if idx < eval_start:
            continue

        if d.get("filter") == "strict-match":
            continue

        # ---- extract CoT ----
        cot = d["resps"][0][0].strip()

        steps = (
            [s.strip() for s in cot.split("\n") if s.strip()]
            if "\n" in cot
            else [s.strip() for s in cot.split(".") if s.strip()]
        )

        if len(steps) == 0:
            continue

        # ---- PRM scoring ----
        try:
            scores = eval_cot_prm(
                d["arguments"]["gen_args_0"]["arg_0"],
                steps
            )
        except Exception as e:
            print(f"[WARN] PRM failed → skip sample {idx}. {e}")
            continue

        if len(scores) == 0:
            continue

        # ---- record raw facts ----
        STEP_TEXTS_ALL.append(steps)
        STEP_SCORES_ALL.append(scores)

        step_token_lens = [
            len(gen_tokenizer.encode(s, add_special_tokens=False))
            for s in steps
        ]
        STEP_TOKEN_LEN.append(step_token_lens)

        Y.append(int(d.get(label, 0)))

    # ============================================================
    # SANITY CHECK
    # ============================================================
    L = len(Y)
    assert all(len(x) == L for x in [
        STEP_SCORES_ALL,
        STEP_TEXTS_ALL,
        STEP_TOKEN_LEN
    ]), "❌ Length mismatch detected!"

    return (
        Y,
        STEP_SCORES_ALL,
        STEP_TEXTS_ALL,
        STEP_TOKEN_LEN
    )
