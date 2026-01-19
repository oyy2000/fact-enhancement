import argparse
import json
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel#, BitsAndBytesConfig
import os
import random
import re  

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
# 🔧 CONFIGURATION & PROMPTS
# ============================================================
DEFAULT_PRM_MODEL = "Qwen/Qwen2.5-Math-PRM-7B"

# 严格遵照官方要求的 System Prompt
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# ============================================================
# ARGPARSE
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", required=True, help="Logical model name (for JSON key)")
parser.add_argument("--gen_model", required=True, help="HF repo id for decoder tokenizer")
parser.add_argument("--layer", required=True)
parser.add_argument("--lam", required=True)
parser.add_argument("--jsonl", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--eval_start", type=int, default=0)
parser.add_argument("--prm_model", default=DEFAULT_PRM_MODEL, help="Path or HF ID for the PRM model")

args = parser.parse_args()

# ============================================================
# LOAD PRM MODEL
# ============================================================
print(f"🔄 Loading PRM Model: {args.prm_model} ...")

prm_tokenizer = AutoTokenizer.from_pretrained(
    args.prm_model, trust_remote_code=True
)

# 注意：针对 72B 模型，这里默认启用了 quantization_config 防止显存爆炸
# 如果你的显存足够（>150GB），可以注释掉 quantization_config
prm_model = AutoModel.from_pretrained(
    args.prm_model,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    # quantization_config=bnb_config, 
    trust_remote_code=True,
).eval()

STEP_TOKEN_ID = prm_tokenizer.encode("<extra_0>")[0]
print("✅ PRM Model Loaded Successfully.")

# ============================================================
# ✅ OFFICIAL SCORING FUNCTION
# ============================================================
def make_step_rewards(logits, token_masks):
    """
    Qwen-Math-PRM 官方提供的评分函数
    只提取 <extra_0> 位置的 positive class 概率
    """
    probabilities = F.softmax(logits, dim=-1)
    # 将非 mask 位置（即非 <extra_0> 位置）的概率置零
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] # seq_len, num_labels
        # 提取非零元素，并取出 label=1 (good) 的概率
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1] 
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res


def eval_cot_prm(query, steps, model, tokenizer, step_token_id):
    """
    构造对话并计算 PRM 分数
    """
    # 构造 step 文本，每一步后面跟一个 <extra_0>
    # 注意：join 后末尾手动补一个 <extra_0>
    text = "<extra_0>".join(steps) + "<extra_0>"
    
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
        {"role": "assistant", "content": text}
    ]
    
    # 应用 chat template
    conv = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False
    )
    
    # 编码 input_ids
    input_ids = tokenizer.encode(conv, return_tensors="pt").to(model.device)

    # 构造 token_masks (定位 <extra_0> 的位置)
    token_masks = (input_ids == step_token_id)

    with torch.no_grad():
        output = model(input_ids=input_ids, use_cache=False)
    
    # 调用官方评分函数
    # output[0] 是 logits
    scores_list = make_step_rewards(output[0], token_masks)
    
    # make_step_rewards 返回的是 list of lists (因为支持 batch)，这里我们只有一个 sample
    return scores_list[0] if scores_list else []


def split_steps_for_qwen(cot_text):
    """
    针对 Qwen 2.5 Math 的切分策略。
    优先尝试 \n\n，如果只有单行，则尝试 \n，
    最后如果是一大坨文本，则尝试按句子切分（作为兜底）。
    """
    if not cot_text:
        return []
        
    cot_text = cot_text.strip()
    
    # 策略 1: 官方推荐的双换行 (Double Newline) - 常见于 Instruct 模型
    if "\n\n" in cot_text:
        return [s.strip() for s in cot_text.split("\n\n") if s.strip()]
    
    # 策略 2: 单换行 (Single Newline) - 常见于简单格式
    if "\n" in cot_text:
        return [s.strip() for s in cot_text.split("\n") if s.strip()]
        
    # 策略 3 (兜底): 如果完全没有换行，尝试按句号切分 (需小心 LaTeX)
    # 这是一个简单的 Regex，避开了数字中的点 (如 3.14)
    # 但无法完美避开 LaTeX 内部，仅作为 fallback
    steps = re.split(r'(?<=[.!?])\s+', cot_text)
    return [s.strip() for s in steps if s.strip()]


# ============================================================
# MAIN DATASET RUNNER
# ============================================================
def run_dataset(jsonl_path, gen_model_name, prm_model, prm_tokenizer, step_token_id, label="exact_match", eval_start=100):
    
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

    print(f"🚀 Starting evaluation from index {eval_start}...")

    for idx, d in enumerate(data):

        if idx < eval_start:
            continue

        # 过滤条件
        if d.get("filter") == "strict-match":
            continue

        # ---- extract CoT ----
        cot = d["resps"][0][0].strip()

        # 简单的分步逻辑 (按换行符)
        steps = split_steps_for_qwen(cot)

        if len(steps) == 0:
            continue

        # ---- PRM scoring ----
        try:
            # 获取原始问题
            query_text = d["arguments"]["gen_args_0"]["arg_0"]
            
            scores = eval_cot_prm(
                query=query_text,
                steps=steps,
                model=prm_model,
                tokenizer=prm_tokenizer,
                step_token_id=step_token_id
            )
        except Exception as e:
            print(f"[WARN] PRM failed → skip sample {idx}. {e}")
            # 出错时可以选择跳过，或者填入空值
            continue

        # 校验步数与分数数量是否一致 (理论上 make_step_rewards 应该返回 len(steps) 个分数)
        # 如果不一致（比如 tokenization 导致 <extra_0> 被吞或者逻辑错误），需要处理
        if len(scores) != len(steps):
             print(f"[WARN] Mismatch: {len(scores)} scores vs {len(steps)} steps. Skipping.")
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
        
        if idx % 10 == 0:
            print(f"Processing {idx}/{len(data)}...", end="\r")

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


# ============================================================
# RUN PRM EVAL
# ============================================================

Y, STEP_SCORES_ALL, STEP_TEXTS_ALL, STEP_TOKEN_LEN = run_dataset(
    jsonl_path=args.jsonl,
    gen_model_name=args.gen_model,
    prm_model=prm_model,          
    prm_tokenizer=prm_tokenizer,  
    step_token_id=STEP_TOKEN_ID,  
    eval_start=args.eval_start,
)

# ============================================================
# SAVE RESULT
# ============================================================

res = {
    args.model_name: {
        args.layer: {
            args.lam: {
                "file_used": args.jsonl,
                "gen_model": args.gen_model,
                "prm_model": args.prm_model, 
                "system_prompt": SYSTEM_PROMPT, # Record prompt used
                "Y": Y,
                "step_scores": STEP_SCORES_ALL,
                "step_texts": STEP_TEXTS_ALL,
                "step_token_len": STEP_TOKEN_LEN
            }
        }
    }
}

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(res, f, indent=2)

print(f"\n✔ Saved chunk → {args.out}")