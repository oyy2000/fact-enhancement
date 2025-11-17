#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# python 00_prepare_pairs.py --input data/GSM8K/test.jsonl --output data/pairs.jsonl

"""
GSM8K -> (question, cot_step1_true, cot_step1_false, error_type)

功能：
1) 从 answer 中抽取 “首步 CoT” （第一句/第一行推理）。
2) 清洗 GSM8K 的 <<...>> 标注，保留可读算式。
3) 在不改变句式风格的前提下制造“首步错误 CoT”，优先：
   - 忽略条件（例如在 a - b - c 中去掉一个减数）
   - 若不适用，则做轻微数值幻觉（把一个数 ±1）
4) 输出 JSONL，每行形如：
   {"question": ..., "cot_step1_true": ..., "cot_step1_false": ..., "error_type": "overlook|hallucinate"}

用法：
python gsm8k_make_pairs.py --input gsm8k.jsonl --output pairs.jsonl
支持 --input 省略（从 stdin 读），也支持单条 JSON 输入。
"""

import argparse, json, re, sys, random
from typing import Tuple, Optional, List

# 允许的安全字符（用于算式提取/计算）
SAFE_EXPR_RE = re.compile(r"^[0-9\+\-\*/\(\)\s]+$")

# 抽取第一行/第一句作为首步（尽量稳健）
def extract_first_step(answer: str) -> str:
    # 取第一条非空行，忽略以 #### 开头的最终答案
    for line in answer.splitlines():
        s = line.strip()
        if not s or s.startswith("####"):
            continue
        # 去除 <<...>> 标注：保留等号右侧数字或直接移除
        s = remove_angle_annotations(s)
        # 若结尾没有句号，补一个以规范化风格
        if not s.endswith((".", "!", "？", "。")):
            s = s.rstrip() + "."
        return s
    # 兜底：整个 answer 清洗后作为一行
    clean = remove_angle_annotations(answer).strip().splitlines()[0]
    return clean if clean else answer.strip()

def remove_angle_annotations(text: str) -> str:
    # GSM8K 中常见 "<<a=b>>b" 或 "<<expr=value>>value"
    # 策略：去掉整个 <<...>>，保留它后面紧随的数字（若有）
    # 例："16 - 3 - 4 = <<16-3-4=9>>9 duck eggs" -> "16 - 3 - 4 = 9 duck eggs"
    text = re.sub(r"<<[^>]*>>\s*", "", text)
    return text

# 从一句话中提取“主算式”（优先选择包含 = 的最右侧算式）
def extract_equation_span(sent: str) -> Optional[Tuple[Tuple[int,int], str]]:
    """
    返回：((start_idx, end_idx), expr_text) 其中 expr_text 是包含等号的子串，或不含等号则返回纯算式
    策略：
      1) 先找带 '=' 的片段（如 "16 - 3 - 4 = 9"）
      2) 没有 '=' 再回退到纯算式（如 "16 - 3 - 4"）
    """
    # 找带等号的片段（尽量短）
    eq_matches = list(re.finditer(r"([0-9][0-9\+\-\*/\s\(\)]*\s*=\s*[0-9\+\-\*/\s\(\)]+)", sent))
    if eq_matches:
        m = eq_matches[-1]  # 使用最右侧的一个（更像“这一步”的最终算式）
        return (m.span(), m.group(1).strip())

    # 退化：寻找纯算式（无等号），尽量选包含运算符的
    expr_matches = list(re.finditer(r"([0-9][0-9\+\-\*/\s\(\)]*[0-9])", sent))
    if expr_matches:
        # 选择含有运算符的候选
        expr_matches = [m for m in expr_matches if re.search(r"[\+\-\*/]", m.group(1))]
        if expr_matches:
            m = expr_matches[-1]
            return (m.span(), m.group(1).strip())
    return None

def safe_eval(expr: str) -> Optional[int]:
    expr = expr.replace(" ", "")
    if not SAFE_EXPR_RE.match(expr):
        return None
    try:
        val = eval(expr, {"__builtins__": {}}, {})
        if isinstance(val, (int, float)):
            # GSM8K 多为整数
            return int(val) if abs(val - int(val)) < 1e-9 else None
    except Exception:
        return None
    return None

def tokenize_minus_chain(expr: str) -> Optional[List[int]]:
    """
    若 expr 形如 a - b - c（允许空格），解析为整数链 [a, b, c]；否则返回 None
    """
    # 只接受减法链（忽略括号/其他运算）
    if re.search(r"[+\*/\(\)]", expr):
        return None
    parts = [p.strip() for p in expr.split("-")]
    try:
        nums = [int(p) for p in parts if p != ""]
        if len(nums) >= 2:
            return nums
    except Exception:
        return None
    return None

def format_minus_chain(nums: List[int]) -> str:
    return " - ".join(str(x) for x in nums)

def make_overlook(expr_text: str) -> Optional[str]:
    """
    忽略条件：对于 a - b - c，把最后一个 -term 去掉，得到 a - b，并重算等号右值
    """
    # 拆分 "LHS = RHS" 或者只有 LHS
    if "=" in expr_text:
        lhs, _rhs = [x.strip() for x in expr_text.split("=", 1)]
    else:
        lhs, _rhs = expr_text.strip(), None

    chain = tokenize_minus_chain(lhs)
    if not chain or len(chain) < 3:
        return None  # 不是 a-b-c 格式，放弃本策略

    new_chain = chain[:-1]  # 去掉最后一个减数
    new_lhs = format_minus_chain(new_chain)
    new_val = safe_eval(new_lhs)
    if new_val is None:
        return None
    return f"{new_lhs} = {new_val}"

def make_hallucinate(expr_text: str) -> Optional[str]:
    """
    轻微数值幻觉：把某个常数 ±1（避免 0/负数不合理时的变化）
    """
    # 分离等号两侧，优先只动 LHS
    if "=" in expr_text:
        lhs, rhs = [x.strip() for x in expr_text.split("=", 1)]
    else:
        lhs, rhs = expr_text.strip(), None

    nums = list(re.finditer(r"\d+", lhs))
    if not nums:
        return None
    # 选择一个非末尾的大于等于 2 的数进行 ±1
    candidates = [m for m in nums if int(m.group(0)) >= 2]
    if not candidates:
        candidates = nums

    m = random.choice(candidates)
    start, end = m.span()
    n = int(m.group(0))
    delta = random.choice([-1, +1])
    new_n = max(0, n + delta)

    new_lhs = lhs[:start] + str(new_n) + lhs[end:]
    # 若有 RHS，则重新计算 RHS，保持“等号”风格一致
    new_val = safe_eval(new_lhs)
    if new_val is not None:
        return f"{new_lhs} = {new_val}"
    else:
        # 计算失败就仅替换 LHS，不给等号
        return new_lhs

def build_false_step(true_step: str) -> Tuple[str, str]:
    """
    输入一句“首步 CoT”，制造一个“句式一致”的错误版本。
    返回 (false_step_text, error_type)
    """
    # 1) 找到句中的主算式
    hit = extract_equation_span(true_step)
    if not hit:
        # 找不到算式时，做轻微数值幻觉：把句中出现的第一个数字 ±1
        fs, et = hallucinate_in_sentence(true_step), "hallucinate"
        return fs, et

    (s, e), eq = hit
    # 2) 优先忽略条件
    over = make_overlook(eq)
    if over:
        false_sent = true_step[:s] + over + true_step[e:]
        return normalize_style(false_sent), "overlook"

    # 3) 退而求其次：数值幻觉
    hal = make_hallucinate(eq)
    if hal:
        false_sent = true_step[:s] + hal + true_step[e:]
        return normalize_style(false_sent), "hallucinate"

    # 4) 实在不行，句内幻觉
    fs, et = hallucinate_in_sentence(true_step), "hallucinate"
    return fs, et

def hallucinate_in_sentence(sent: str) -> str:
    # 把句中的第一个数字 ±1（保持句式不变）
    m = re.search(r"\d+", sent)
    if not m:
        return sent  # 没数字就不改
    start, end = m.span()
    n = int(m.group(0))
    delta = -1 if n >= 2 else +1
    new_n = max(0, n + delta)
    return normalize_style(sent[:start] + str(new_n) + sent[end:])

def normalize_style(s: str) -> str:
    # 去多空白，保证句尾有句号
    s = re.sub(r"\s+", " ", s).strip()
    if not s.endswith((".", "!", "？", "。")):
        s += "."
    return s

def process_record(obj: dict) -> Optional[dict]:
    q = obj.get("question", "").strip()
    a = obj.get("answer", "").strip()
    if not q or not a:
        return None
    true_step = extract_first_step(a)
    false_step, etype = build_false_step(true_step)

    # 轻微风格对齐（长度靠拢）
    tgt_len = max(len(true_step), 40)
    def style_pad(x: str) -> str:
        if len(x) >= tgt_len:
            return x
        return x + " " + ("Therefore. " * 10)
    true_step = normalize_style(style_pad(true_step))[:tgt_len]
    false_step = normalize_style(style_pad(false_step))[:tgt_len]

    return {
        "question": q,
        "cot_step1_true": true_step,
        "cot_step1_false": false_step,
        "error_type": etype
    }

def iter_inputs(fp):
    # 既支持 JSONL，也支持单条 JSON
    txt = fp.read().strip()
    if not txt:
        return
    # 判断是否多行
    if "\n" in txt:
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
    else:
        # 单行 JSON
        yield json.loads(txt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="-", help="JSONL 文件；- 代表从 stdin 读")
    ap.add_argument("--output", type=str, required=True, help="输出 JSONL")
    args = ap.parse_args()

    fin = sys.stdin if args.input == "-" else open(args.input, "r", encoding="utf-8")
    fout = open(args.output, "w", encoding="utf-8")

    cnt = 0
    for obj in iter_inputs(fin):
        rec = process_record(obj)
        if rec:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cnt += 1
    fout.close()
    if fin is not sys.stdin:
        fin.close()
    print(f"[OK] written {cnt} pairs to {args.output}")

if __name__ == "__main__":
    random.seed(42)
    main()
