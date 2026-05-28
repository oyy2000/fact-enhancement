#!/usr/bin/env python3
"""
Find samples that flip from wrong (baseline) → correct (steered),
AND have increased ρ (tok/step) AND decreased step count.
"""

import json
from pathlib import Path
from transformers import AutoTokenizer

BASE = Path("/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement")
EXP  = BASE / "exps/gpt_rewrites_unified_new/Qwen_Qwen2.5-3B-Instruct/vectors_50_old/Qwen_Qwen2.5-3B-Instruct_applied/gsm8k_cot_zeroshot_unified_selected_layers"

BL_JSONL = list((EXP / "Qwen2.5-3B-Instruct_L6_BASELINE" / "Qwen__Qwen2.5-3B-Instruct").glob("samples_*.jsonl"))[0]
ST_JSONL = list((EXP / "Qwen2.5-3B-Instruct_L6_lam4p0"   / "Qwen__Qwen2.5-3B-Instruct").glob("samples_*.jsonl"))[0]

FILTER = "flexible-extract"   # use the flexible filter
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# ── helpers ─────────────────────────────────────────────────────────────────
def load_samples(path):
    """Return dict  doc_id → record  (only the chosen filter)."""
    out = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("filter") == FILTER:
                out[rec["doc_id"]] = rec
    return out

def get_text(rec):
    resps = rec.get("resps", [[]])
    return resps[0][0] if resps and resps[0] else ""

def compute_rho_and_steps(text, tokenizer):
    steps = [s.strip() for s in (text or "").split("\n\n") if s.strip()]
    n_steps = len(steps)
    if n_steps == 0:
        return 0.0, 0
    total_toks = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in steps)
    rho = total_toks / n_steps
    return rho, n_steps

# ── main ────────────────────────────────────────────────────────────────────
print("Loading tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

print(f"Loading baseline:  {BL_JSONL.name}")
bl = load_samples(BL_JSONL)
print(f"Loading steered:   {ST_JSONL.name}")
st = load_samples(ST_JSONL)

common_ids = sorted(set(bl) & set(st))
print(f"Common doc_ids (filter={FILTER}): {len(common_ids)}")

# ── scan ────────────────────────────────────────────────────────────────────
hits = []
for did in common_ids:
    b, s = bl[did], st[did]
    # wrong → correct
    if b.get("exact_match", 0) != 0.0 or s.get("exact_match", 0) != 1.0:
        continue
    b_text = get_text(b)
    s_text = get_text(s)
    b_rho, b_steps = compute_rho_and_steps(b_text, tokenizer)
    s_rho, s_steps = compute_rho_and_steps(s_text, tokenizer)
    # rho increased AND steps decreased
    if s_rho > b_rho and s_steps < b_steps:
        hits.append({
            "doc_id": did,
            "question": b["doc"]["question"][:120],
            "target": b["target"].split("####")[-1].strip() if "####" in b["target"] else b["target"][-20:],
            "bl_rho": round(b_rho, 2),
            "st_rho": round(s_rho, 2),
            "delta_rho": round(s_rho - b_rho, 2),
            "bl_steps": b_steps,
            "st_steps": s_steps,
            "bl_text": b_text,
            "st_text": s_text,
        })

print(f"\n{'='*80}")
print(f"Wrong→Correct  AND  ρ↑  AND  steps↓ :  {len(hits)}  samples")
print(f"{'='*80}\n")

for i, h in enumerate(hits):
    print(f"── [{i+1}] doc_id={h['doc_id']}  answer={h['target']}  ──")
    print(f"   Question: {h['question']}…")
    print(f"   Baseline:  ρ={h['bl_rho']:.1f}  steps={h['bl_steps']}  (WRONG)")
    print(f"   Steered:   ρ={h['st_rho']:.1f}  steps={h['st_steps']}  (CORRECT)")
    print(f"   Δρ = +{h['delta_rho']:.1f},  Δsteps = {h['st_steps'] - h['bl_steps']}")
    print()

# ── save full details ───────────────────────────────────────────────────────
out_path = BASE / "documents" / "flipped_rho_up_steps_down.json"
# save without the full text for the JSON (too large), but print a few examples
save_list = []
for h in hits:
    save_list.append({k: v for k, v in h.items()})
with open(out_path, "w") as f:
    json.dump(save_list, f, indent=2, ensure_ascii=False)
print(f"\nFull details saved → {out_path}")

# ── print 3 detailed examples ──────────────────────────────────────────────
print(f"\n{'='*80}")
print("DETAILED EXAMPLES (first 3)")
print(f"{'='*80}")
for h in hits[:3]:
    print(f"\n{'─'*60}")
    print(f"doc_id={h['doc_id']}  |  answer={h['target']}")
    print(f"Question: {h['question']}")
    print(f"\n[BASELINE]  ρ={h['bl_rho']:.1f}  steps={h['bl_steps']}  WRONG")
    print(h["bl_text"][:600])
    print(f"\n[STEERED]   ρ={h['st_rho']:.1f}  steps={h['st_steps']}  CORRECT")
    print(h["st_text"][:600])
