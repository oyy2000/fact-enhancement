#!/usr/bin/env python3
"""
Generate a summary table (CSV + PNG) for all Qwen models:
  - #Samples, #Degenerate, %Degenerate
  - Avg tokens/step BEFORE filtering
  - Avg tokens/step AFTER removing degenerate samples
  - Avg total_tokens BEFORE / AFTER
Save to new_exps/figures/figure1/
"""

import json, re, os
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────
DATA_DIR = Path("new_exps/figure1_sampling_data")
OUT_DIR  = Path("new_exps/figures/figure1")
DATASET  = "gsm8k_samples.jsonl"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Same thresholds as analyze_repetition.py
BIGRAM_REP_THRESH = 0.5
TRIGRAM_REP_THRESH = 0.4
SENTENCE_DUP_THRESH = 0.4
LONG_REPEAT_CHAR_THRESH = 80

# ── Helpers (same as analyze_repetition.py) ─────────────────────────────
def tokenize_simple(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())

def ngram_repetition_rate(tokens, n):
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(ngrams)

def sentence_duplication_rate(text):
    sents = [s.strip() for s in re.split(r'[.\n]', text) if len(s.strip()) > 10]
    if len(sents) <= 1:
        return 0.0
    seen = set()
    dups = 0
    for s in sents:
        if s in seen:
            dups += 1
        seen.add(s)
    return dups / len(sents)

def longest_repeated_substring_len(text, min_len=20):
    text = text[:3000]
    n = len(text)
    if n < min_len * 2:
        return 0
    MOD = (1 << 61) - 1
    BASE = 131

    def has_repeat(length):
        seen = {}
        h = 0
        power = pow(BASE, length, MOD)
        for i in range(n):
            h = (h * BASE + ord(text[i])) % MOD
            if i >= length:
                h = (h - ord(text[i - length]) * power) % MOD
            if i >= length - 1:
                if h in seen:
                    for prev_i in seen[h]:
                        if text[prev_i - length + 1:prev_i + 1] == text[i - length + 1:i + 1]:
                            return True
                    seen[h].append(i)
                else:
                    seen[h] = [i]
        return False

    lo, hi = min_len, min(n // 2, 300)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if has_repeat(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best

def is_degenerate(resp, tokens=None):
    if tokens is None:
        tokens = tokenize_simple(resp)
    bigram_rep = ngram_repetition_rate(tokens, 2)
    trigram_rep = ngram_repetition_rate(tokens, 3)
    sent_dup = sentence_duplication_rate(resp)
    lrs_len = longest_repeated_substring_len(resp) if len(tokens) > 50 else 0
    return (bigram_rep > BIGRAM_REP_THRESH
            or trigram_rep > TRIGRAM_REP_THRESH
            or sent_dup > SENTENCE_DUP_THRESH
            or lrs_len > LONG_REPEAT_CHAR_THRESH)

# ── Main ────────────────────────────────────────────────────────────────
def process_model(model_dir):
    fpath = model_dir / DATASET
    if not fpath.exists():
        return None

    all_tps = []       # avg_tokens_per_step for ALL samples
    all_tok = []       # total_tokens for ALL samples
    clean_tps = []     # avg_tokens_per_step for non-degenerate
    clean_tok = []     # total_tokens for non-degenerate
    n_degen = 0
    n_total = 0

    with open(fpath) as f:
        for line in f:
            row = json.loads(line)
            for sample in row["samples"]:
                n_total += 1
                tps = sample["avg_tokens_per_step"]
                tok = sample["total_tokens"]
                all_tps.append(tps)
                all_tok.append(tok)

                resp = sample["response"]
                if is_degenerate(resp):
                    n_degen += 1
                else:
                    clean_tps.append(tps)
                    clean_tok.append(tok)

    return {
        "model": model_dir.name,
        "n_total": n_total,
        "n_degen": n_degen,
        "pct_degen": 100 * n_degen / n_total if n_total else 0,
        "avg_tps_before": np.mean(all_tps),
        "avg_tps_after": np.mean(clean_tps) if clean_tps else 0,
        "avg_tok_before": np.mean(all_tok),
        "avg_tok_after": np.mean(clean_tok) if clean_tok else 0,
        "med_tps_before": np.median(all_tps),
        "med_tps_after": np.median(clean_tps) if clean_tps else 0,
        "med_tok_before": np.median(all_tok),
        "med_tok_after": np.median(clean_tok) if clean_tok else 0,
    }


def size_sort_key(name):
    """Extract numeric size for sorting: 0.5B -> 0.5, 1.5B -> 1.5, etc."""
    m = re.search(r'(\d+\.?\d*)B', name)
    return float(m.group(1)) if m else 999


def main():
    model_dirs = sorted(DATA_DIR.iterdir())
    # Only Qwen models
    model_dirs = [d for d in model_dirs if d.is_dir() and 'Qwen' in d.name and (d / DATASET).exists()]
    model_dirs.sort(key=lambda d: size_sort_key(d.name))

    print(f"Processing {len(model_dirs)} Qwen models...")
    results = []
    for md in model_dirs:
        print(f"  {md.name} ...", end=" ", flush=True)
        r = process_model(md)
        if r:
            results.append(r)
            print(f"done ({r['n_degen']}/{r['n_total']} degenerate)")

    # ── Print table ─────────────────────────────────────────────────────
    print("\n")
    header = (f"{'Model':<35} {'#Total':>7} {'#Degen':>7} {'%Degen':>7} "
              f"{'AvgTPS':>8} {'AvgTPS*':>8} {'Δ':>6} "
              f"{'AvgTok':>8} {'AvgTok*':>8} {'Δ':>6}")
    print(header)
    print("-" * len(header))
    for r in results:
        short = r["model"].replace("Qwen_Qwen2.5-", "Qwen2.5-")
        delta_tps = r["avg_tps_after"] - r["avg_tps_before"]
        delta_tok = r["avg_tok_after"] - r["avg_tok_before"]
        print(f"{short:<35} {r['n_total']:>7} {r['n_degen']:>7} {r['pct_degen']:>6.1f}% "
              f"{r['avg_tps_before']:>8.1f} {r['avg_tps_after']:>8.1f} {delta_tps:>+6.1f} "
              f"{r['avg_tok_before']:>8.1f} {r['avg_tok_after']:>8.1f} {delta_tok:>+6.1f}")
    print("\n* = after removing degenerate samples")

    # ── Save CSV ────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "repetition_summary_qwen.csv"
    with open(csv_path, "w") as f:
        f.write("Model,#Total,#Degenerate,%Degenerate,"
                "AvgTokensPerStep_Before,AvgTokensPerStep_After,ΔTPS,"
                "AvgTotalTokens_Before,AvgTotalTokens_After,ΔTok,"
                "MedTokensPerStep_Before,MedTokensPerStep_After,"
                "MedTotalTokens_Before,MedTotalTokens_After\n")
        for r in results:
            short = r["model"].replace("Qwen_Qwen2.5-", "Qwen2.5-")
            f.write(f"{short},{r['n_total']},{r['n_degen']},{r['pct_degen']:.1f},"
                    f"{r['avg_tps_before']:.2f},{r['avg_tps_after']:.2f},"
                    f"{r['avg_tps_after']-r['avg_tps_before']:.2f},"
                    f"{r['avg_tok_before']:.2f},{r['avg_tok_after']:.2f},"
                    f"{r['avg_tok_after']-r['avg_tok_before']:.2f},"
                    f"{r['med_tps_before']:.2f},{r['med_tps_after']:.2f},"
                    f"{r['med_tok_before']:.2f},{r['med_tok_after']:.2f}\n")
    print(f"\nCSV saved to: {csv_path}")

    # ── Plot table as image ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.axis("off")

    col_labels = [
        "Model", "#Samples", "#Degen", "%Degen",
        "Avg TPS\n(before)", "Avg TPS\n(after)", "ΔTPS",
        "Avg Tok\n(before)", "Avg Tok\n(after)", "ΔTok"
    ]
    cell_data = []
    for r in results:
        short = r["model"].replace("Qwen_Qwen2.5-", "Qwen2.5-")
        cell_data.append([
            short,
            f"{r['n_total']}",
            f"{r['n_degen']}",
            f"{r['pct_degen']:.1f}%",
            f"{r['avg_tps_before']:.1f}",
            f"{r['avg_tps_after']:.1f}",
            f"{r['avg_tps_after']-r['avg_tps_before']:+.1f}",
            f"{r['avg_tok_before']:.1f}",
            f"{r['avg_tok_after']:.1f}",
            f"{r['avg_tok_after']-r['avg_tok_before']:+.1f}",
        ])

    table = ax.table(cellText=cell_data, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight 0.5B row (row index 1 since 0 is header)
    for j in range(len(col_labels)):
        table[1, j].set_facecolor("#FFF2CC")

    # Color the delta columns
    for i, r in enumerate(results):
        delta_tps = r['avg_tps_after'] - r['avg_tps_before']
        delta_tok = r['avg_tok_after'] - r['avg_tok_before']
        # Negative delta = good (fewer tokens after filtering)
        color_tps = "#C6EFCE" if delta_tps < -1 else "#FFFFFF"
        color_tok = "#C6EFCE" if delta_tok < -5 else "#FFFFFF"
        table[i+1, 6].set_facecolor(color_tps)
        table[i+1, 9].set_facecolor(color_tok)

    ax.set_title("Qwen2.5 Models: Repetition / Degenerate Output Analysis (GSM8K)\n"
                 "TPS = Avg Tokens Per Step  |  Tok = Avg Total Tokens  |  * = after removing degenerate samples",
                 fontsize=11, fontweight="bold", pad=20)

    fig.tight_layout()
    png_path = OUT_DIR / "repetition_summary_qwen.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Table image saved to: {png_path}")


if __name__ == "__main__":
    main()
