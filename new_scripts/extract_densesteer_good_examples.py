#!/usr/bin/env python3
"""
Extract DenseSteer examples where density increased or step count decreased.
Outputs a JSON with full before/after text + audit metrics for qualitative analysis.

Usage:
    python new_scripts/extract_densesteer_good_examples.py
"""

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# --- paths ---
REWRITE_JSON = BASE / "exps" / "gpt_rewrites_unified_new" / "Qwen_Qwen2.5-3B-Instruct" / "rewritten_old.json"
AUDIT_JSON   = BASE / "new_exps" / "e3" / "audit_densesteer_old.json"
OUT_DIR      = BASE / "new_exps" / "e3"

def main():
    # load source data (keyed by doc_id)
    with open(REWRITE_JSON) as f:
        raw = json.load(f)
    src_by_id = {entry["doc_id"]: entry for entry in raw}
    print(f"Loaded {len(raw)} source entries from rewritten_old.json")

    # load audit
    with open(AUDIT_JSON) as f:
        audit = json.load(f)
    per_sample = audit["per_sample"]
    print(f"Loaded {len(per_sample)} audit entries")

    # --- filter: density_delta > 0 OR step_count_delta < 0 ---
    good = []
    for s in per_sample:
        a = s["auto"]
        dens_up   = a["density_delta"] > 0
        step_down = a["step_count_delta"] < 0
        if dens_up or step_down:
            doc_id = s["doc_id"]
            src = src_by_id.get(doc_id, {})
            entry = {
                "doc_id": doc_id,
                "question": src.get("doc", {}).get("question", ""),
                "target": src.get("target", ""),
                "resp_before": src.get("resp_before", ""),
                "resp_after": src.get("resp_after", ""),
                "steps_before": a["step_count_before"],
                "steps_after": a["step_count_after"],
                "step_delta": a["step_count_delta"],
                "density_before": a["density_before"],
                "density_after": a["density_after"],
                "density_delta": a["density_delta"],
                "token_overlap": a["token_overlap_jaccard"],
                "edit_similarity": a["edit_similarity"],
                "category": (
                    "density_up_and_step_down" if dens_up and step_down
                    else "density_up" if dens_up
                    else "step_down"
                ),
            }
            # attach judge if available
            if "judge" in s:
                entry["judge_majority"] = s["judge"].get("majority", {})
            good.append(entry)

    # sort by density_delta descending (best improvements first)
    good.sort(key=lambda x: x["density_delta"], reverse=True)

    # --- stats ---
    n_both = sum(1 for g in good if g["category"] == "density_up_and_step_down")
    n_dens = sum(1 for g in good if g["category"] == "density_up")
    n_step = sum(1 for g in good if g["category"] == "step_down")
    print(f"\nFiltered: {len(good)}/{len(per_sample)} samples")
    print(f"  density_up AND step_down: {n_both}")
    print(f"  density_up only:          {n_dens}")
    print(f"  step_down only:           {n_step}")

    # --- save full JSON ---
    out_full = OUT_DIR / "densesteer_good_examples.json"
    with open(out_full, "w", encoding="utf-8") as f:
        json.dump(good, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(good)} examples -> {out_full}")

    # --- save a readable markdown summary of top examples ---
    out_md = OUT_DIR / "densesteer_good_examples_top20.md"
    top_n = min(20, len(good))
    lines = [f"# DenseSteer: Top {top_n} Examples (density↑ or steps↓)\n"]
    for i, g in enumerate(good[:top_n]):
        lines.append(f"## Example {i+1} — doc_id={g['doc_id']}  [{g['category']}]")
        lines.append(f"- Steps: {g['steps_before']} → {g['steps_after']} (Δ={g['step_delta']:+d})")
        lines.append(f"- Density: {g['density_before']:.2f} → {g['density_after']:.2f} (Δ={g['density_delta']:+.2f})")
        lines.append(f"- Token overlap: {g['token_overlap']:.4f}, Edit sim: {g['edit_similarity']:.4f}")
        if g.get("judge_majority"):
            jm = g["judge_majority"]
            lines.append(f"- Judge: meaning_preserved={jm.get('reasoning_meaning_preserved')}, "
                         f"final_answer={jm.get('final_answer_preserved')}, "
                         f"new_facts={jm.get('new_facts_introduced')}")
        lines.append(f"\n**Question:** {g['question'][:200]}{'...' if len(g['question'])>200 else ''}")
        # show first 500 chars of before/after
        lines.append(f"\n**Before** ({g['steps_before']} steps, ρ={g['density_before']:.1f}):")
        lines.append(f"```\n{g['resp_before'][:500]}{'...' if len(g['resp_before'])>500 else ''}\n```")
        lines.append(f"\n**After** ({g['steps_after']} steps, ρ={g['density_after']:.1f}):")
        lines.append(f"```\n{g['resp_after'][:500]}{'...' if len(g['resp_after'])>500 else ''}\n```")
        lines.append("---\n")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved top-{top_n} readable summary -> {out_md}")


if __name__ == "__main__":
    main()
