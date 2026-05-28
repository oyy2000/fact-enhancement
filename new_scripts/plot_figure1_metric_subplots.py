#!/usr/bin/env python3
"""Plot multiple rho-like metrics vs GSM8K accuracy in one subplot grid.

Default input is the DAS-enriched jsonl from add_das_to_figure1_sampling.py.
The default aggregation mirrors 05_plots_figure1.py:
  - drop degenerate samples if --filter-degenerate is set
  - y-axis accuracy is computed over all retained samples
  - x-axis metrics use only correct samples if --correct-only is set
  - question-level means are averaged by default
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

DATA_DIR = Path("new_exps/figure1_sampling_data")
OUT_DIR = Path("new_exps/figures/figure1")

MODELS = [
    ("Qwen_Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B-Instruct", 0.5, "o"),
    ("Qwen_Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B-Instruct", 1.5, "s"),
    ("Qwen_Qwen2.5-3B-Instruct", "Qwen2.5-3B-Instruct", 3, "^"),
    ("Qwen_Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct", 7, "D"),
    ("Qwen_Qwen2.5-14B-Instruct", "Qwen2.5-14B-Instruct", 14, "P"),
    ("Qwen_Qwen2.5-32B-Instruct", "Qwen2.5-32B-Instruct", 32, "X"),
    ("Qwen_Qwen2.5-72B-Instruct", "Qwen2.5-72B-Instruct", 72, "*"),
]

MODEL_COLORS = [
    "#1f77b4", "#2ca02c", "#8c564b", "#7f7f7f",
    "#17becf", "#9467bd", "#d62728",
]

BIGRAM_REP_THRESH = 0.5
TRIGRAM_REP_THRESH = 0.4
SENTENCE_DUP_THRESH = 0.4
LONG_REPEAT_CHAR_THRESH = 80

TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[A-Za-z]+|[+\-*/=<>≤≥%$]")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "he", "her", "his", "if", "in", "is", "it", "of",
    "on", "or", "she", "that", "the", "then", "there", "they", "this",
    "to", "we", "will", "with", "you",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--input-name", default="gsm8k_samples_with_das.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR / "figure1_gsm8k_rholike_metric_subplots_filtered_correct_only.png",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["rho", "log_rho", "nll", "das", "unique_rho", "content_rho"],
        choices=("rho", "log_rho", "nll", "das", "unique_rho", "content_rho"),
    )
    parser.add_argument("--correct-only", action="store_true", default=True)
    parser.add_argument("--all-samples-x", action="store_true")
    parser.add_argument("--filter-degenerate", action="store_true", default=True)
    parser.add_argument("--no-filter-degenerate", action="store_true")
    parser.add_argument("--bigram-thresh", type=float, default=BIGRAM_REP_THRESH)
    parser.add_argument("--trigram-thresh", type=float, default=TRIGRAM_REP_THRESH)
    parser.add_argument("--sentdup-thresh", type=float, default=SENTENCE_DUP_THRESH)
    parser.add_argument("--long-repeat-thresh", type=int, default=LONG_REPEAT_CHAR_THRESH)
    parser.add_argument("--x-aggregation", choices=("question", "sample"), default="question")
    parser.add_argument("--exclude-models", nargs="*", default=[])
    parser.add_argument("--no-corr-line", action="store_true")
    args = parser.parse_args()
    if args.all_samples_x:
        args.correct_only = False
    if args.no_filter_degenerate:
        args.filter_degenerate = False
    return args


def normalize_model_key(value) -> str:
    return (
        str(value).lower()
        .replace("qwen/", "")
        .replace("_", "")
        .replace("-", "")
        .replace(".", "p")
        .replace(" ", "")
    )


def model_is_excluded(folder, label, params_b, excluded) -> bool:
    keys = {
        normalize_model_key(folder),
        normalize_model_key(label),
        normalize_model_key(f"{params_b:g}B"),
        normalize_model_key(params_b),
    }
    return any(normalize_model_key(item) in keys for item in excluded)


def tokenize_simple(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


def ngram_repetition_rate(tokens: list[str], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(ngrams)


def sentence_duplication_rate(text: str) -> float:
    sents = [s.strip() for s in re.split(r"[.\n]", text) if len(s.strip()) > 10]
    if len(sents) <= 1:
        return 0.0
    seen = set()
    dups = 0
    for sent in sents:
        if sent in seen:
            dups += 1
        seen.add(sent)
    return dups / len(sents)


def has_repeated_substring(text: str, length: int) -> bool:
    text = text[:3000]
    n = len(text)
    if length <= 0 or n < length * 2:
        return False
    mod = (1 << 61) - 1
    base = 131
    seen = {}
    h = 0
    power = pow(base, length, mod)
    for i, ch in enumerate(text):
        h = (h * base + ord(ch)) % mod
        if i >= length:
            h = (h - ord(text[i - length]) * power) % mod
        if i >= length - 1:
            if h in seen:
                start = i - length + 1
                cur = text[start:i + 1]
                for prev_i in seen[h]:
                    prev_start = prev_i - length + 1
                    if text[prev_start:prev_i + 1] == cur:
                        return True
                seen[h].append(i)
            else:
                seen[h] = [i]
    return False


def is_degenerate_sample(sample: dict, args: argparse.Namespace) -> bool:
    resp = sample.get("response", "")
    tokens = tokenize_simple(resp)
    return (
        ngram_repetition_rate(tokens, 2) > args.bigram_thresh
        or ngram_repetition_rate(tokens, 3) > args.trigram_thresh
        or sentence_duplication_rate(resp) > args.sentdup_thresh
        or (
            len(tokens) > 50
            and args.long_repeat_thresh >= 0
            and has_repeated_substring(resp, args.long_repeat_thresh + 1)
        )
    )


def sem(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.array(values, dtype=float)
    return float(np.std(arr) / np.sqrt(len(arr)))


def sample_nll(sample: dict) -> float:
    for key in ("nll", "mean_nll", "response_nll"):
        if key in sample and sample[key] is not None:
            return float(sample[key])
    if "ppl" in sample and sample["ppl"] is not None:
        return math.log(float(sample["ppl"]))
    raise KeyError("Missing per-sample NLL/PPL. Use gsm8k_samples_with_das.jsonl.")


def response_tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def content_tokens(text: str) -> list[str]:
    toks = response_tokens(text)
    out = []
    for tok in toks:
        if re.fullmatch(r"\d+(?:\.\d+)?", tok):
            out.append(tok)
        elif tok in {"+", "-", "*", "/", "=", "<", ">", "≤", "≥", "%", "$"}:
            out.append(tok)
        elif tok.isalpha() and tok not in STOPWORDS and len(tok) > 1:
            out.append(tok)
    return out


def sample_metric(sample: dict, metric: str) -> float:
    rho = float(sample["density_rho"])
    if metric == "rho":
        return rho
    if metric == "log_rho":
        return math.log(rho)
    if metric == "nll":
        return sample_nll(sample)
    if metric == "das":
        if "das" in sample and sample["das"] is not None:
            return float(sample["das"])
        return math.log(rho) - sample_nll(sample)

    n_steps = max(float(sample["n_steps"]), 1.0)
    if metric == "unique_rho":
        toks = [t for t in response_tokens(sample.get("response", "")) if re.search(r"\w", t)]
        return len(set(toks)) / n_steps
    if metric == "content_rho":
        return len(content_tokens(sample.get("response", ""))) / n_steps
    raise ValueError(metric)


METRIC_LABELS = {
    "rho": r"$\rho$ (tokens/step)",
    "log_rho": r"$\log\rho$",
    "nll": "NLL",
    "das": r"DAS = $\log\rho$ - NLL",
    "unique_rho": "Unique-token density",
    "content_rho": "Content-token density",
}


def load_model_rows(folder: str, args: argparse.Namespace) -> dict | None:
    path = args.data_dir / folder / args.input_name
    if not path.is_file():
        return None

    q_accs: list[float] = []
    q_metric_values = {metric: [] for metric in args.metrics}
    pooled_metric_values = {metric: [] for metric in args.metrics}
    n_total = 0
    n_retained = 0
    n_metric_samples = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            samples = rec["samples"]
            n_total += len(samples)
            if args.filter_degenerate:
                samples = [s for s in samples if not is_degenerate_sample(s, args)]
            if not samples:
                continue
            n_retained += len(samples)
            q_accs.append(float(np.mean([bool(s["correct"]) for s in samples]) * 100))

            metric_samples = [s for s in samples if s["correct"]] if args.correct_only else samples
            if not metric_samples:
                continue
            n_metric_samples += len(metric_samples)
            for metric in args.metrics:
                vals = [sample_metric(s, metric) for s in metric_samples]
                q_metric_values[metric].append(float(np.mean(vals)))
                pooled_metric_values[metric].extend(vals)

    if not q_accs:
        return None

    row = {
        "acc": float(np.mean(q_accs)),
        "acc_sem": sem(q_accs),
        "n_total": n_total,
        "n_retained": n_retained,
        "n_metric_samples": n_metric_samples,
        "n_questions_acc": len(q_accs),
    }
    for metric in args.metrics:
        vals = pooled_metric_values[metric] if args.x_aggregation == "sample" else q_metric_values[metric]
        if vals:
            row[f"avg_{metric}"] = float(np.mean(vals))
            row[f"{metric}_sem"] = sem(vals)
            row[f"n_questions_{metric}"] = len(q_metric_values[metric])
        else:
            row[f"avg_{metric}"] = float("nan")
            row[f"{metric}_sem"] = float("nan")
            row[f"n_questions_{metric}"] = 0
    return row


def finite_xy(xs, ys):
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if not pairs:
        return np.array([]), np.array([])
    x_arr, y_arr = zip(*pairs)
    return np.array(x_arr), np.array(y_arr)


def format_p_value(p: float) -> str:
    if not math.isfinite(p):
        return "p = NA"
    if p < 0.001:
        return "p < 0.001"
    if p < 0.01:
        return f"p = {p:.3f}"
    return f"p = {p:.2f}"


def add_corr(ax, xs, ys) -> dict:
    xs, ys = finite_xy(xs, ys)
    out = {"n": len(xs), "pearson_r": float("nan"), "p_value": float("nan")}
    if len(xs) < 2 or np.std(xs) == 0 or np.std(ys) == 0:
        ax.text(0.04, 0.96, f"r = NA\nn = {len(xs)}", transform=ax.transAxes,
                ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8))
        return out
    r, p = pearsonr(xs, ys)
    slope, intercept = np.polyfit(xs, ys, 1)
    pad = (xs.max() - xs.min()) * 0.06
    x_fit = np.linspace(xs.min() - pad, xs.max() + pad, 100)
    ax.plot(x_fit, slope * x_fit + intercept, "--", color="black", lw=1.1, alpha=0.6)
    ax.text(0.04, 0.96, f"r = {r:.3f}\n{format_p_value(float(p))}\nn = {len(xs)}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8))
    out.update({"pearson_r": float(r), "p_value": float(p), "slope": float(slope), "intercept": float(intercept)})
    return out


def write_csvs(rows: list[dict], corr_rows: list[dict], output: Path, metrics: list[str]) -> None:
    csv_path = output.with_suffix(".csv")
    fieldnames = [
        "model", "acc", "acc_sem", "n_total", "n_retained",
        "n_metric_samples", "n_questions_acc",
    ]
    for metric in metrics:
        fieldnames.extend([f"avg_{metric}", f"{metric}_sem", f"n_questions_{metric}"])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    corr_path = output.with_name(output.stem + "_corr.csv")
    corr_fields = ["metric", "n", "pearson_r", "p_value", "slope", "intercept"]
    with open(corr_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=corr_fields)
        writer.writeheader()
        for row in corr_rows:
            writer.writerow({k: row.get(k, "") for k in corr_fields})
    print(f"Saved: {csv_path}")
    print(f"Saved: {corr_path}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.output.is_absolute():
        args.output = args.out_dir / args.output.name

    rows = []
    for idx, (folder, label, params_b, marker) in enumerate(MODELS):
        if model_is_excluded(folder, label, params_b, args.exclude_models):
            continue
        row = load_model_rows(folder, args)
        if row is None:
            continue
        row.update({
            "folder": folder,
            "model": label,
            "params_b": params_b,
            "marker": marker,
            "color": MODEL_COLORS[idx],
        })
        rows.append(row)

    if not rows:
        raise SystemExit("No rows loaded.")

    n_metrics = len(args.metrics)
    n_cols = 3 if n_metrics > 2 else n_metrics
    n_rows = int(math.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.6 * n_rows), squeeze=False)
    axes_flat = list(axes.ravel())

    corr_rows = []
    for ax, metric in zip(axes_flat, args.metrics):
        xs = [row[f"avg_{metric}"] for row in rows]
        ys = [row["acc"] for row in rows]
        xerrs = [row[f"{metric}_sem"] for row in rows]
        yerrs = [row["acc_sem"] for row in rows]

        for row, x, y, xe, ye in zip(rows, xs, ys, xerrs, yerrs):
            ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="none", ecolor=row["color"],
                        elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=2)
            ax.scatter(x, y, c=row["color"], marker=row["marker"], s=95,
                       edgecolors="none", label=row["model"], zorder=3)

        if not args.no_corr_line:
            corr = add_corr(ax, xs, ys)
            corr["metric"] = metric
            corr_rows.append(corr)
        ax.set_title(METRIC_LABELS[metric], fontsize=13, fontweight="bold")
        ax.set_xlabel(METRIC_LABELS[metric], fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(alpha=0.3)

    for ax in axes_flat[n_metrics:]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 1.02))
    title_bits = []
    if args.filter_degenerate:
        title_bits.append("filtered")
    title_bits.append("correct-only x" if args.correct_only else "all-sample x")
    fig.suptitle("GSM8K rho-like metrics vs accuracy (" + ", ".join(title_bits) + ")",
                 fontsize=14, fontweight="bold", y=1.055)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")
    write_csvs(rows, corr_rows, args.output, args.metrics)


if __name__ == "__main__":
    main()
