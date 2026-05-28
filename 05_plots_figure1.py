#!/usr/bin/env python3
"""Figure 1 - GSM8K: Steps & Reasoning Density/DAS/NLL vs Accuracy.

Data source: new_exps/figure1_sampling_data/Qwen_*/gsm8k_samples.jsonl
Two panels:
  (a) Avg Steps  vs  Accuracy   + error bars (SEM)
  (b) Avg metric vs Accuracy + error bars (SEM)
Use --corr-line to add least-squares auxiliary lines and Pearson r/p/n labels.
Legend on top, distinct markers per model, no annotations on points.
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ── style ──
FONT_SIZE = 20

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": 14,
    "font.family": "sans-serif",
})

def apply_plot_style(ax, *, fontsize=FONT_SIZE):
    ax.set_title("")
    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.grid(alpha=0.3)


def draw_sem_errorbars(ax, x, y, *, xerr, yerr, color):
    """Draw horizontal metric SEM and emphasize vertical accuracy SEM."""
    ax.errorbar(
        x, y, xerr=xerr,
        fmt="none", ecolor=color, elinewidth=1.2,
        capsize=3, capthick=1.2, alpha=0.75, zorder=2,
    )
    ax.errorbar(
        x, y, yerr=yerr,
        fmt="none", ecolor="#222222", elinewidth=1.8,
        capsize=5, capthick=1.8, alpha=0.95, zorder=2.8,
    )

# ── paths ──
DATA_DIR = Path("new_exps/figure1_sampling_data")
OUT_DIR  = Path("new_exps/figures/figure1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Qwen2.5 models with distinct markers (matching reference style) ──
MODELS = [
    ("Qwen_Qwen2.5-0.5B-Instruct",  "Qwen2.5-0.5B-Instruct",  0.5,  "o"),   # circle
    ("Qwen_Qwen2.5-1.5B-Instruct",  "Qwen2.5-1.5B-Instruct",  1.5,  "s"),   # square
    ("Qwen_Qwen2.5-3B-Instruct",    "Qwen2.5-3B-Instruct",     3,    "^"),   # triangle up
    ("Qwen_Qwen2.5-7B-Instruct",    "Qwen2.5-7B-Instruct",     7,    "D"),   # diamond
    ("Qwen_Qwen2.5-14B-Instruct",   "Qwen2.5-14B-Instruct",    14,   "P"),   # plus (filled)
    ("Qwen_Qwen2.5-32B-Instruct",   "Qwen2.5-32B-Instruct",    32,   "X"),   # x (filled)
    ("Qwen_Qwen2.5-72B-Instruct",   "Qwen2.5-72B-Instruct",    72,   "*"),   # star
]

# Distinct colors matching reference image palette
MODEL_COLORS = [
    "#1f77b4",  # 0.5B - blue
    "#2ca02c",  # 1.5B - green
    "#8c564b",  # 3B   - brown
    "#7f7f7f",  # 7B   - gray
    "#17becf",  # 14B  - cyan
    "#9467bd",  # 32B  - purple
    "#d62728",  # 72B  - red
]

# Same default heuristic as new_scripts/repetition_table.py.
BIGRAM_REP_THRESH = 0.5
TRIGRAM_REP_THRESH = 0.4
SENTENCE_DUP_THRESH = 0.4
LONG_REPEAT_CHAR_THRESH = 80


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--input-name",
        default="gsm8k_samples.jsonl",
        help="Filename under each model folder. Use an enriched jsonl for DAS.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output png path. If omitted, a descriptive name is used.",
    )
    parser.add_argument(
        "--metric",
        choices=("rho", "log_rho", "total_tokens", "nll", "das"),
        default="rho",
        help="Bottom-panel x metric. NLL/DAS require per-sample NLL/PPL/DAS fields.",
    )
    parser.add_argument(
        "--correct-only",
        action="store_true",
        help="Compute x-axis metrics using only correct samples. Accuracy is still computed over all retained samples.",
    )
    parser.add_argument(
        "--filter-degenerate",
        action="store_true",
        help="Drop repeated/degenerate samples before computing accuracy and x metrics.",
    )
    parser.add_argument("--bigram-thresh", type=float, default=BIGRAM_REP_THRESH)
    parser.add_argument("--trigram-thresh", type=float, default=TRIGRAM_REP_THRESH)
    parser.add_argument("--sentdup-thresh", type=float, default=SENTENCE_DUP_THRESH)
    parser.add_argument("--long-repeat-thresh", type=int, default=LONG_REPEAT_CHAR_THRESH)
    parser.add_argument(
        "--x-aggregation",
        choices=("question", "sample"),
        default="question",
        help="Average x metrics by question first (original behavior) or over pooled samples.",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Write a sidecar CSV with the plotted values.",
    )
    parser.add_argument(
        "--corr-line",
        action="store_true",
        help="Add a least-squares auxiliary line and Pearson r/p/n annotation to each panel.",
    )
    parser.add_argument(
        "--no-corr-text",
        action="store_true",
        help="With --corr-line, draw the auxiliary line but hide the Pearson annotation box.",
    )
    parser.add_argument(
        "--exclude-models",
        nargs="*",
        default=[],
        help="Model folders, labels, or size strings to exclude, e.g. 0.5B Qwen2.5-0.5B-Instruct.",
    )
    return parser.parse_args()


def normalize_model_key(value):
    return (
        str(value)
        .lower()
        .replace("qwen/", "")
        .replace("_", "")
        .replace("-", "")
        .replace(".", "p")
        .replace(" ", "")
    )


def model_is_excluded(folder, label, params_b, excluded):
    keys = {
        normalize_model_key(folder),
        normalize_model_key(label),
        normalize_model_key(f"{params_b:g}B"),
        normalize_model_key(f"{params_b:g}b"),
        normalize_model_key(params_b),
    }
    return any(normalize_model_key(item) in keys for item in excluded)


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


def has_repeated_substring(text, length):
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


def is_degenerate_sample(sample, args):
    resp = sample.get("response", "")
    tokens = tokenize_simple(resp)
    bigram_rep = ngram_repetition_rate(tokens, 2)
    trigram_rep = ngram_repetition_rate(tokens, 3)
    sent_dup = sentence_duplication_rate(resp)
    has_long_repeat = (
        has_repeated_substring(resp, args.long_repeat_thresh + 1)
        if len(tokens) > 50 and args.long_repeat_thresh >= 0
        else False
    )
    return (
        bigram_rep > args.bigram_thresh
        or trigram_rep > args.trigram_thresh
        or sent_dup > args.sentdup_thresh
        or has_long_repeat
    )


def sample_nll(sample):
    for key in ("nll", "mean_nll", "response_nll"):
        if key in sample:
            return float(sample[key])
    if "ppl" in sample:
        return math.log(float(sample["ppl"]))
    raise KeyError(
        "NLL/DAS needs per-sample 'nll'/'mean_nll'/'response_nll', or 'ppl'. "
        "Use gsm8k_samples_with_das.jsonl or run new_scripts/add_das_to_figure1_sampling.py first."
    )


def sample_bottom_metric(sample, metric):
    rho = float(sample["density_rho"])
    if metric == "rho":
        return rho
    if metric == "log_rho":
        return math.log(rho)
    if metric == "total_tokens":
        if "total_tokens" in sample:
            return float(sample["total_tokens"])
        return rho * float(sample["n_steps"])
    if metric == "nll":
        return sample_nll(sample)

    if "das" in sample:
        return float(sample["das"])
    if "DAS" in sample:
        return float(sample["DAS"])

    nll = sample_nll(sample)
    return math.log(rho) - nll


def sem(values):
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return np.nan
    return float(np.std(arr) / np.sqrt(len(arr)))


def load_gsm8k_stats(folder: str, args):
    p = args.data_dir / folder / args.input_name
    if not p.exists():
        return None
    q_accs, q_steps, q_metrics = [], [], []
    pooled_steps, pooled_metrics = [], []
    n_total, n_retained, n_correct_metric = 0, 0, 0
    with open(p) as f:
        for line in f:
            rec = json.loads(line)
            samples = rec["samples"]
            n_total += len(samples)
            if args.filter_degenerate:
                samples = [s for s in samples if not is_degenerate_sample(s, args)]
            if not samples:
                continue
            n_retained += len(samples)

            cc = [bool(s["correct"]) for s in samples]
            q_accs.append(np.mean(cc) * 100)
            metric_samples = [s for s in samples if s["correct"]] if args.correct_only else samples
            if not metric_samples:
                continue
            if args.correct_only:
                n_correct_metric += len(metric_samples)
            steps = [float(s["n_steps"]) for s in metric_samples]
            metrics = [sample_bottom_metric(s, args.metric) for s in metric_samples]
            q_steps.append(np.mean(steps))
            q_metrics.append(np.mean(metrics))
            pooled_steps.extend(steps)
            pooled_metrics.extend(metrics)

    if not q_accs or not q_steps or not q_metrics:
        return None
    if args.x_aggregation == "sample":
        avg_steps = float(np.mean(pooled_steps))
        steps_sem = sem(pooled_steps)
        avg_metric = float(np.mean(pooled_metrics))
        metric_sem = sem(pooled_metrics)
    else:
        avg_steps = float(np.mean(q_steps))
        steps_sem = sem(q_steps)
        avg_metric = float(np.mean(q_metrics))
        metric_sem = sem(q_metrics)

    n = len(q_accs)
    return {
        "acc": float(np.mean(q_accs)),
        "acc_sem": float(np.std(q_accs) / np.sqrt(n)),
        "avg_steps": avg_steps,
        "steps_sem": steps_sem,
        "avg_metric": avg_metric,
        "metric_sem": metric_sem,
        "n_questions_acc": len(q_accs),
        "n_questions_metric": len(q_steps),
        "n_total": n_total,
        "n_retained": n_retained,
        "n_metric_samples": len(pooled_steps),
        "n_correct_metric": n_correct_metric,
    }


def finite_xy(xs, ys):
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if not pairs:
        return np.array([], dtype=float), np.array([], dtype=float)
    x_arr, y_arr = zip(*pairs)
    return np.array(x_arr, dtype=float), np.array(y_arr, dtype=float)


def format_p_value(p):
    if not math.isfinite(p):
        return "p = NA"
    if p < 0.001:
        return "p < 0.001"
    if p < 0.01:
        return f"p = {p:.3f}"
    return f"p = {p:.2f}"


def add_corr_line_and_annotation(ax, xs, ys, *, loc="lower left", show_text=True):
    xs, ys = finite_xy(xs, ys)
    stats = {
        "n": len(xs),
        "pearson_r": float("nan"),
        "p_value": float("nan"),
        "slope": float("nan"),
        "intercept": float("nan"),
    }

    can_fit = len(xs) >= 2 and np.std(xs) > 0 and np.std(ys) > 0
    if can_fit:
        r, p = pearsonr(xs, ys)
        slope, intercept = np.polyfit(xs, ys, deg=1)
        stats.update({
            "pearson_r": float(r),
            "p_value": float(p),
            "slope": float(slope),
            "intercept": float(intercept),
        })

        pad = (xs.max() - xs.min()) * 0.06
        x_fit = np.linspace(xs.min() - pad, xs.max() + pad, 100)
        y_fit = slope * x_fit + intercept
        ax.plot(
            x_fit,
            y_fit,
            linestyle="--",
            color="black",
            linewidth=1.4,
            alpha=0.65,
            zorder=1,
        )
        text = f"Pearson r = {r:.3f}\n{format_p_value(float(p))}\nn = {len(xs)}"
    else:
        text = f"Pearson r = NA\nn = {len(xs)}"

    if show_text:
        props = dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8)
        pos = {"lower left": (0.05, 0.05, "left", "bottom"),
               "lower right": (0.95, 0.05, "right", "bottom"),
               "upper left": (0.05, 0.95, "left", "top"),
               "upper right": (0.95, 0.95, "right", "top")}
        x, y, ha, va = pos.get(loc, pos["lower left"])
        ax.text(x, y, text, transform=ax.transAxes, fontsize=16,
                verticalalignment=va, horizontalalignment=ha, bbox=props)
    return stats


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 12))

    rows = []

    for i, (folder, label, params_b, marker) in enumerate(MODELS):
        if model_is_excluded(folder, label, params_b, args.exclude_models):
            continue
        try:
            stats = load_gsm8k_stats(folder, args)
        except KeyError as exc:
            raise SystemExit(str(exc)) from exc
        if stats is None:
            continue
        acc = stats["acc"]
        acc_sem = stats["acc_sem"]
        avg_steps = stats["avg_steps"]
        steps_sem = stats["steps_sem"]
        avg_metric = stats["avg_metric"]
        metric_sem = stats["metric_sem"]
        rows.append({"model": label, **stats})

        color = MODEL_COLORS[i]
        ms = 120

        # Panel (a)
        draw_sem_errorbars(ax1, avg_steps, acc, xerr=steps_sem, yerr=acc_sem, color=color)
        ax1.scatter(avg_steps, acc, c=color, marker=marker, s=ms,
                    edgecolors="white", linewidths=0.6, zorder=3, label=label)

        # Panel (b)
        draw_sem_errorbars(ax2, avg_metric, acc, xerr=metric_sem, yerr=acc_sem, color=color)
        ax2.scatter(avg_metric, acc, c=color, marker=marker, s=ms,
                    edgecolors="white", linewidths=0.6, zorder=3, label=label)

    if not rows:
        raise SystemExit("No model rows available after filtering/exclusion.")

    ax1.set_xlabel("Avg Number of Steps")
    ax1.set_ylabel("Accuracy")
    apply_plot_style(ax1)

    if args.metric == "rho":
        ax2.set_xlabel(r"Avg Reasoning Density $\rho$ (tokens/step)")
    elif args.metric == "log_rho":
        ax2.set_xlabel(r"Avg $\log\rho$")
    elif args.metric == "total_tokens":
        ax2.set_xlabel("Avg Total Tokens")
    elif args.metric == "nll":
        ax2.set_xlabel("Avg NLL")
    else:
        ax2.set_xlabel(r"Avg DAS $= \log\rho - \mathrm{NLL}$")
    ax2.set_ylabel("Accuracy")
    apply_plot_style(ax2)

    corr_rows = []
    if args.corr_line:
        step_stats = add_corr_line_and_annotation(
            ax1,
            [row["avg_steps"] for row in rows],
            [row["acc"] for row in rows],
            loc="lower left",
            show_text=not args.no_corr_text,
        )
        step_stats.update({"panel": "steps", "x_metric": "avg_steps", "y_metric": "acc"})
        corr_rows.append(step_stats)

        metric_stats = add_corr_line_and_annotation(
            ax2,
            [row["avg_metric"] for row in rows],
            [row["acc"] for row in rows],
            loc="lower left",
            show_text=not args.no_corr_text,
        )
        metric_stats.update({"panel": args.metric, "x_metric": f"avg_{args.metric}", "y_metric": "acc"})
        corr_rows.append(metric_stats)

        for corr in corr_rows:
            print(
                f"Pearson {corr['x_metric']} vs {corr['y_metric']}: "
                f"r={corr['pearson_r']:.6g}, p={corr['p_value']:.6g}, n={corr['n']}"
            )

    # Shared legend on top
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=2, frameon=False, fontsize=14,
               bbox_to_anchor=(0.5, 1.04), markerscale=1.2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if args.output is None:
        suffix = []
        if args.filter_degenerate:
            suffix.append("filtered")
        if args.correct_only:
            suffix.append("correct_only")
        if args.metric == "das":
            suffix.append("das")
        if args.x_aggregation == "sample":
            suffix.append("sample_weighted")
        if args.corr_line:
            suffix.append("corr_line")
        if args.no_corr_text:
            suffix.append("no_corr_text")
        tail = "_" + "_".join(suffix) if suffix else ""
        out = args.out_dir / f"figure1_gsm8k_steps_{args.metric}{tail}.png"
    else:
        out = args.output
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    if args.write_csv:
        csv_path = out.with_suffix(".csv")
        header = [
            "model", "acc", "acc_sem", "avg_steps", "steps_sem",
            f"avg_{args.metric}", f"{args.metric}_sem",
            "n_questions_acc", "n_questions_metric", "n_total",
            "n_retained", "n_metric_samples",
        ]
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for row in rows:
                values = [
                    row["model"],
                    f"{row['acc']:.6g}",
                    f"{row['acc_sem']:.6g}",
                    f"{row['avg_steps']:.6g}",
                    f"{row['steps_sem']:.6g}",
                    f"{row['avg_metric']:.6g}",
                    f"{row['metric_sem']:.6g}",
                    str(row["n_questions_acc"]),
                    str(row["n_questions_metric"]),
                    str(row["n_total"]),
                    str(row["n_retained"]),
                    str(row["n_metric_samples"]),
                ]
                f.write(",".join(values) + "\n")
        print(f"Saved: {csv_path}")

        if corr_rows:
            corr_csv_path = out.with_name(out.stem + "_corr.csv")
            corr_header = [
                "panel", "x_metric", "y_metric", "n",
                "pearson_r", "p_value", "slope", "intercept",
            ]
            with open(corr_csv_path, "w", encoding="utf-8") as f:
                f.write(",".join(corr_header) + "\n")
                for corr in corr_rows:
                    values = [
                        corr["panel"],
                        corr["x_metric"],
                        corr["y_metric"],
                        str(corr["n"]),
                        f"{corr['pearson_r']:.8g}",
                        f"{corr['p_value']:.8g}",
                        f"{corr['slope']:.8g}",
                        f"{corr['intercept']:.8g}",
                    ]
                    f.write(",".join(values) + "\n")
            print(f"Saved: {corr_csv_path}")


if __name__ == "__main__":
    main()
