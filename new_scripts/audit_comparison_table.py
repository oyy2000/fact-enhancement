#!/usr/bin/env python3
"""
Build a side-by-side comparison table from multiple audit_results.json files.
Outputs CSV + prints a formatted table.

Usage:
    python new_scripts/audit_comparison_table.py \
        --audits new_exps/e3/audit_densesteer_old.json \
                 new_exps/e3/audit_e3_rule_based.json \
                 new_exps/e3/audit_e3_2_gpt54mini.json \
        --labels "DenseSteer (GPT-5.1)" "E3 Rule-Based" "E3.2 GPT-5-mini" \
        --output new_exps/e3/audit_comparison.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path


JUDGE_KEYS = [
    "final_answer_preserved",
    "reasoning_meaning_preserved",
    "new_facts_introduced",
    "error_fixed",
    "mainly_adjacent_merge",
    "style_preserved",
]


def load_report(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["report"]


def build_rows(reports, labels):
    """Yield (metric_name, val_1, val_2, ...) tuples."""
    # --- header ---
    yield ["Metric"] + labels

    # n_samples
    yield ["n_samples"] + [str(r["n_samples"]) for r in reports]

    # step counts: before -> after
    yield ["Steps before (mean)"] + [
        f"{r['auto_metrics']['step_count_before_mean']:.1f}" for r in reports
    ]
    yield ["Steps after (mean)"] + [
        f"{r['auto_metrics']['step_count_after_mean']:.1f}" for r in reports
    ]

    # density: before -> after
    yield ["Density before (mean)"] + [
        f"{r['auto_metrics']['density_before_mean']:.1f}" for r in reports
    ]
    yield ["Density after (mean)"] + [
        f"{r['auto_metrics']['density_after_mean']:.1f}" for r in reports
    ]

    # other auto metrics
    for key in ["token_overlap_jaccard", "edit_similarity",
                "adjacent_merge_ratio", "changed_nums_ops_ratio"]:
        yield [key + " (mean±std)"] + [
            f"{r['auto_metrics'][key]['mean']:.4f}±{r['auto_metrics'][key]['std']:.4f}"
            for r in reports
        ]

    # separator
    yield ["--- GPT Judge ---"] + [""] * len(reports)

    # judge metrics
    for key in JUDGE_KEYS:
        vals = []
        for r in reports:
            jm = r.get("judge_metrics", {})
            if key in jm:
                vals.append(f"{jm[key]['true_pct']:.1f}% (agree {jm[key]['agreement_pct']:.0f}%)")
            else:
                vals.append("N/A")
        yield [key] + vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audits", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--output", default="audit_comparison.csv")
    args = ap.parse_args()

    assert len(args.audits) == len(args.labels), "audits and labels must match"

    reports = [load_report(p) for p in args.audits]
    rows = list(build_rows(reports, args.labels))

    # --- print ---
    col_widths = [max(len(str(row[c])) for row in rows) for c in range(len(rows[0]))]
    for i, row in enumerate(rows):
        line = "  ".join(str(row[c]).ljust(col_widths[c]) for c in range(len(row)))
        print(line)
        if i == 0:
            print("  ".join("-" * w for w in col_widths))

    # --- save CSV ---
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    print(f"\nSaved to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
