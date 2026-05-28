#!/usr/bin/env python3
"""
Analyze repetition / degenerate output in figure1 sampling data.
Specifically motivated by reviewer concern: 0.5B model has excessively many
tokens per step — is this due to repetition / output collapse?

Checks performed per sample:
  1. N-gram repetition rate (bigram, trigram, 4-gram)
  2. Sentence-level repetition (exact duplicate sentences)
  3. Longest repeated substring (detects single-word / phrase loops)
  4. Token count vs step count scatter stats
  5. Flags "degenerate" samples (heuristic thresholds)

Usage:
  python new_scripts/analyze_repetition.py
"""

import json, os, re, sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────
DATA_DIR = Path("new_exps/figure1_sampling_data")
DATASET = "gsm8k_samples.jsonl"

# Heuristic thresholds for "degenerate"
BIGRAM_REP_THRESH = 0.5       # >50% bigrams are repeated
TRIGRAM_REP_THRESH = 0.4
SENTENCE_DUP_THRESH = 0.4     # >40% sentences are duplicates
LONG_REPEAT_CHAR_THRESH = 80  # longest repeated substring > 80 chars


# ── Helpers ─────────────────────────────────────────────────────────────
def tokenize_simple(text: str) -> list[str]:
    """Whitespace + punctuation tokenizer (good enough for repetition check)."""
    return re.findall(r"\w+|[^\w\s]", text.lower())


def ngram_repetition_rate(tokens: list[str], n: int) -> float:
    """Fraction of n-grams that appear more than once."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(ngrams)


def sentence_duplication_rate(text: str) -> tuple[float, list[str]]:
    """Fraction of sentences that are exact duplicates of an earlier sentence."""
    # Split on period / newline
    sents = [s.strip() for s in re.split(r'[.\n]', text) if len(s.strip()) > 10]
    if len(sents) <= 1:
        return 0.0, []
    seen = set()
    dups = []
    for s in sents:
        if s in seen:
            dups.append(s)
        seen.add(s)
    return len(dups) / len(sents), dups


def longest_repeated_substring(text: str, min_len: int = 20) -> str:
    """Find a long repeated substring using binary search + rolling hash (fast)."""
    text = text[:3000]  # cap input for speed
    n = len(text)
    if n < min_len * 2:
        return ""

    def has_repeat(length):
        """Check if any substring of given length appears twice."""
        seen = {}
        MOD = (1 << 61) - 1
        BASE = 131
        h = 0
        power = pow(BASE, length, MOD)
        for i in range(n):
            h = (h * BASE + ord(text[i])) % MOD
            if i >= length:
                h = (h - ord(text[i - length]) * power) % MOD
            if i >= length - 1:
                if h in seen:
                    # Verify to avoid hash collision
                    for prev_i in seen[h]:
                        if text[prev_i - length + 1:prev_i + 1] == text[i - length + 1:i + 1]:
                            return text[i - length + 1:i + 1]
                    seen[h].append(i)
                else:
                    seen[h] = [i]
        return None

    # Binary search for longest length
    lo, hi = min_len, min(n // 2, 300)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        result = has_repeat(mid)
        if result:
            best = result
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def find_word_loops(text: str, min_repeats: int = 5) -> list[tuple[str, int]]:
    """Detect single-word or short-phrase loops like 'the the the the...'"""
    loops = []
    # Single word repeated
    for m in re.finditer(r'\b(\w{2,})((?:\s+\1){' + str(min_repeats-1) + r',})', text, re.IGNORECASE):
        word = m.group(1)
        count = len(re.findall(r'\b' + re.escape(word) + r'\b', m.group(0), re.IGNORECASE))
        loops.append((word, count))
    # Short phrase repeated (2-4 words)
    for m in re.finditer(r'((?:\w+\s+){1,3}\w+)((?:\s+\1){' + str(min_repeats-1) + r',})', text, re.IGNORECASE):
        phrase = m.group(1).strip()
        count = m.group(0).count(phrase)
        loops.append((phrase, count))
    return loops


# ── Main analysis ───────────────────────────────────────────────────────
def analyze_model(model_dir: Path) -> dict:
    """Analyze all samples for one model."""
    fpath = model_dir / DATASET
    if not fpath.exists():
        return None

    model_name = model_dir.name
    all_stats = []
    degenerate_examples = []

    with open(fpath) as f:
        for line_idx, line in enumerate(f):
            row = json.loads(line)
            doc_id = row["doc_id"]
            for sample in row["samples"]:
                resp = sample["response"]
                tokens = tokenize_simple(resp)
                n_tokens_simple = len(tokens)

                bigram_rep = ngram_repetition_rate(tokens, 2)
                trigram_rep = ngram_repetition_rate(tokens, 3)
                fourgram_rep = ngram_repetition_rate(tokens, 4)
                sent_dup_rate, dup_sents = sentence_duplication_rate(resp)
                word_loops = find_word_loops(resp)
                lrs = longest_repeated_substring(resp) if n_tokens_simple > 50 else ""

                is_degenerate = (
                    bigram_rep > BIGRAM_REP_THRESH
                    or trigram_rep > TRIGRAM_REP_THRESH
                    or sent_dup_rate > SENTENCE_DUP_THRESH
                    or len(lrs) > LONG_REPEAT_CHAR_THRESH
                    or len(word_loops) > 0
                )

                stat = {
                    "doc_id": doc_id,
                    "sample_idx": sample["sample_idx"],
                    "n_steps": sample["n_steps"],
                    "total_tokens": sample["total_tokens"],
                    "avg_tokens_per_step": sample["avg_tokens_per_step"],
                    "n_tokens_simple": n_tokens_simple,
                    "response_chars": len(resp),
                    "bigram_rep": bigram_rep,
                    "trigram_rep": trigram_rep,
                    "fourgram_rep": fourgram_rep,
                    "sent_dup_rate": sent_dup_rate,
                    "longest_repeat_len": len(lrs),
                    "word_loops": len(word_loops),
                    "is_degenerate": is_degenerate,
                    "correct": sample["correct"],
                }
                all_stats.append(stat)

                if is_degenerate:
                    degenerate_examples.append({
                        **stat,
                        "response_preview": resp[:600],
                        "longest_repeat_preview": lrs[:200],
                        "dup_sents_preview": dup_sents[:3],
                        "word_loops_detail": word_loops[:5],
                    })

    return {
        "model": model_name,
        "stats": all_stats,
        "degenerate_examples": degenerate_examples,
    }


def print_summary(result: dict):
    """Print a concise summary for one model."""
    stats = result["stats"]
    n = len(stats)
    degen = [s for s in stats if s["is_degenerate"]]
    model = result["model"]

    print(f"\n{'='*80}")
    print(f"Model: {model}  |  Total samples: {n}  |  Degenerate: {len(degen)} ({100*len(degen)/n:.1f}%)")
    print(f"{'='*80}")

    # Basic token stats
    toks = [s["total_tokens"] for s in stats]
    avg_tps = [s["avg_tokens_per_step"] for s in stats]
    print(f"  total_tokens:       mean={np.mean(toks):.1f}  median={np.median(toks):.1f}  "
          f"p90={np.percentile(toks,90):.1f}  max={np.max(toks)}")
    print(f"  avg_tokens/step:    mean={np.mean(avg_tps):.1f}  median={np.median(avg_tps):.1f}  "
          f"p90={np.percentile(avg_tps,90):.1f}  max={np.max(avg_tps):.1f}")

    # Repetition stats
    bi = [s["bigram_rep"] for s in stats]
    tri = [s["trigram_rep"] for s in stats]
    sd = [s["sent_dup_rate"] for s in stats]
    lr = [s["longest_repeat_len"] for s in stats]
    wl = [s["word_loops"] for s in stats]
    print(f"  bigram_rep_rate:    mean={np.mean(bi):.3f}  p90={np.percentile(bi,90):.3f}  max={np.max(bi):.3f}")
    print(f"  trigram_rep_rate:   mean={np.mean(tri):.3f}  p90={np.percentile(tri,90):.3f}  max={np.max(tri):.3f}")
    print(f"  sent_dup_rate:      mean={np.mean(sd):.3f}  p90={np.percentile(sd,90):.3f}  max={np.max(sd):.3f}")
    print(f"  longest_repeat_len: mean={np.mean(lr):.1f}  p90={np.percentile(lr,90):.1f}  max={np.max(lr)}")
    print(f"  samples_with_word_loops: {sum(1 for w in wl if w > 0)}")

    # Degenerate vs non-degenerate token comparison
    if degen:
        degen_tps = [s["avg_tokens_per_step"] for s in degen]
        normal = [s for s in stats if not s["is_degenerate"]]
        normal_tps = [s["avg_tokens_per_step"] for s in normal]
        print(f"\n  ** Degenerate samples avg_tokens/step: mean={np.mean(degen_tps):.1f}  "
              f"median={np.median(degen_tps):.1f}")
        if normal_tps:
            print(f"  ** Normal samples avg_tokens/step:     mean={np.mean(normal_tps):.1f}  "
                  f"median={np.median(normal_tps):.1f}")

        # Show a few examples
        print(f"\n  Top degenerate examples (sorted by bigram_rep):")
        top = sorted(result["degenerate_examples"], key=lambda x: x["bigram_rep"], reverse=True)[:5]
        for i, ex in enumerate(top):
            print(f"\n  --- Example {i+1}: doc_id={ex['doc_id']}, sample_idx={ex['sample_idx']}")
            print(f"      total_tokens={ex['total_tokens']}, avg_tokens/step={ex['avg_tokens_per_step']:.1f}")
            print(f"      bigram_rep={ex['bigram_rep']:.3f}, trigram_rep={ex['trigram_rep']:.3f}, "
                  f"sent_dup={ex['sent_dup_rate']:.3f}")
            if ex["word_loops_detail"]:
                print(f"      word_loops: {ex['word_loops_detail']}")
            if ex["longest_repeat_preview"]:
                print(f"      longest_repeat: \"{ex['longest_repeat_preview']}\"")
            # Show response snippet
            print(f"      response: \"{ex['response_preview'][:300]}...\"")


def main():
    model_dirs = sorted(DATA_DIR.iterdir())
    model_dirs = [d for d in model_dirs if d.is_dir() and (d / DATASET).exists()]

    print(f"Found {len(model_dirs)} models with {DATASET}")
    print(f"Models: {[d.name for d in model_dirs]}")

    all_results = {}
    for md in model_dirs:
        print(f"\nAnalyzing {md.name} ...")
        result = analyze_model(md)
        if result:
            all_results[md.name] = result
            print_summary(result)

    # ── Cross-model comparison table ────────────────────────────────────
    print(f"\n\n{'='*100}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'='*100}")
    header = f"{'Model':<45} {'#Samp':>6} {'#Degen':>7} {'%Degen':>7} {'AvgTok':>7} {'AvgTPS':>7} {'BiRep':>7} {'TriRep':>7} {'SentDup':>8}"
    print(header)
    print("-" * len(header))
    for name in sorted(all_results.keys()):
        r = all_results[name]
        stats = r["stats"]
        n = len(stats)
        nd = sum(1 for s in stats if s["is_degenerate"])
        avg_tok = np.mean([s["total_tokens"] for s in stats])
        avg_tps = np.mean([s["avg_tokens_per_step"] for s in stats])
        avg_bi = np.mean([s["bigram_rep"] for s in stats])
        avg_tri = np.mean([s["trigram_rep"] for s in stats])
        avg_sd = np.mean([s["sent_dup_rate"] for s in stats])
        print(f"{name:<45} {n:>6} {nd:>7} {100*nd/n:>6.1f}% {avg_tok:>7.1f} {avg_tps:>7.1f} {avg_bi:>7.3f} {avg_tri:>7.3f} {avg_sd:>8.3f}")

    # ── Save detailed degenerate examples to file ───────────────────────
    out_path = DATA_DIR / "repetition_analysis.json"
    save_data = {}
    for name, r in all_results.items():
        save_data[name] = {
            "n_samples": len(r["stats"]),
            "n_degenerate": sum(1 for s in r["stats"] if s["is_degenerate"]),
            "degenerate_pct": 100 * sum(1 for s in r["stats"] if s["is_degenerate"]) / len(r["stats"]),
            "mean_avg_tokens_per_step": float(np.mean([s["avg_tokens_per_step"] for s in r["stats"]])),
            "mean_bigram_rep": float(np.mean([s["bigram_rep"] for s in r["stats"]])),
            "mean_trigram_rep": float(np.mean([s["trigram_rep"] for s in r["stats"]])),
            "mean_sent_dup_rate": float(np.mean([s["sent_dup_rate"] for s in r["stats"]])),
            "degenerate_examples": r["degenerate_examples"][:20],  # top 20
        }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {out_path}")


if __name__ == "__main__":
    main()
