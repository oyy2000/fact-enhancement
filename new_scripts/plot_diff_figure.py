#!/usr/bin/env python3
"""
Side-by-side diff figure: Before (left) vs After (right).
Changed lines get coloured backgrounds; unchanged lines stay white.

Usage:
    python new_scripts/plot_diff_figure.py \
        --input new_exps/e3/densesteer_good_examples_correct50.json \
        --doc_id 1268
"""

import argparse
import difflib
import json
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch


# ── Colours ──────────────────────────────────────────────────────────────
BG_DEL   = "#FEE2E2"   # red highlight for deleted / changed lines (left)
BG_ADD   = "#DCFCE7"   # green highlight for added / changed lines (right)
BG_EQ    = "#FFFFFF"
TX_NORM  = "#1F2937"
TX_DEL   = "#991B1B"
TX_ADD   = "#166534"
TX_GREY  = "#9CA3AF"
BORDER   = "#E5E7EB"
WRAP_W   = 68


def clean(text: str) -> str:
    text = re.sub(r'\\\[', ' ', text)
    text = re.sub(r'\\\]', ' ', text)
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\\(', '', text)
    text = re.sub(r'\\\)', '', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\\boxed\{([^}]*)\}', r'[\1]', text)
    text = re.sub(r'\\times', '×', text)
    text = re.sub(r'\\text', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def wrap_lines(text: str, w: int = WRAP_W):
    out = []
    for line in text.split('\n'):
        if len(line) <= w:
            out.append(line)
        else:
            out.extend(textwrap.wrap(line, width=w, subsequent_indent='    '))
    return out


def align_sides(b_lines, a_lines):
    """
    Use SequenceMatcher to produce aligned (left, right, tag) rows.
    tag: 'equal', 'delete', 'insert', 'replace'
    Blank strings are used as padding so both columns have the same height.
    """
    sm = difflib.SequenceMatcher(None, b_lines, a_lines)
    rows = []  # list of (left_text, right_text, tag)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal':
            for k in range(i2 - i1):
                rows.append((b_lines[i1 + k], a_lines[j1 + k], 'equal'))
        elif op == 'replace':
            n_b = i2 - i1
            n_a = j2 - j1
            n_max = max(n_b, n_a)
            for k in range(n_max):
                lt = b_lines[i1 + k] if k < n_b else ''
                rt = a_lines[j1 + k] if k < n_a else ''
                rows.append((lt, rt, 'replace'))
        elif op == 'delete':
            for k in range(i2 - i1):
                rows.append((b_lines[i1 + k], '', 'delete'))
        elif op == 'insert':
            for k in range(j2 - j1):
                rows.append(('', a_lines[j1 + k], 'insert'))
    return rows


def make_figure(before: str, after: str, meta: dict, out_path: str):
    b_lines = wrap_lines(clean(before))
    a_lines = wrap_lines(clean(after))
    rows = align_sides(b_lines, a_lines)

    n_rows = len(rows)
    fs = 7.0           # font size
    line_h_pt = fs * 1.6
    dpi = 200

    # figure sizing
    header_h = 1.2   # inches for title area
    footer_h = 0.6
    body_h = max(3.0, n_rows * (line_h_pt / 72) + 0.3)
    fig_h = header_h + body_h + footer_h
    fig_w = 15

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(fig_w, fig_h),
        gridspec_kw={'wspace': 0.03, 'left': 0.02, 'right': 0.98,
                     'top': 1 - header_h / fig_h,
                     'bottom': footer_h / fig_h})

    # ── Title ────────────────────────────────────────────────────────────
    title = f"DenseSteer Rewrite — doc_id {meta['doc_id']}"
    sub = (f"Steps: {meta['steps_before']} → {meta['steps_after']}    "
           f"ρ: {meta['density_before']:.1f} → {meta['density_after']:.1f} "
           f"(Δ={meta['density_delta']:+.1f})    "
           f"Token overlap: {meta['token_overlap']:.2f}")
    q = meta['question']
    if len(q) > 140:
        q = q[:137] + '...'

    fig.text(0.5, 1 - 0.25 / fig_h, title, fontsize=13, fontweight='bold',
             ha='center', va='top')
    fig.text(0.5, 1 - 0.55 / fig_h, sub, fontsize=9, ha='center', va='top',
             color='#6B7280')
    fig.text(0.5, 1 - 0.82 / fig_h, f"Q: {q}", fontsize=8, ha='center',
             va='top', color='#4B5563', style='italic')

    # ── Render panels ────────────────────────────────────────────────────
    for ax, side in [(ax_l, 'left'), (ax_r, 'right')]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, n_rows)
        ax.invert_yaxis()
        ax.axis('off')

        # panel title
        if side == 'left':
            ptitle = f"Before  ({meta['steps_before']} steps, ρ = {meta['density_before']:.1f})"
            ax.set_title(ptitle, fontsize=10, fontweight='bold', color=TX_DEL, pad=8)
        else:
            ptitle = f"After  ({meta['steps_after']} steps, ρ = {meta['density_after']:.1f})"
            ax.set_title(ptitle, fontsize=10, fontweight='bold', color=TX_ADD, pad=8)

        for i, (lt, rt, tag) in enumerate(rows):
            text = lt if side == 'left' else rt

            # decide background & text colour
            if tag == 'equal':
                bg = BG_EQ
                fg = TX_NORM
            elif tag == 'replace':
                bg = BG_DEL if side == 'left' else BG_ADD
                fg = TX_DEL if side == 'left' else TX_ADD
            elif tag == 'delete':
                bg = BG_DEL if side == 'left' else BG_EQ
                fg = TX_DEL if side == 'left' else TX_GREY
            elif tag == 'insert':
                bg = BG_ADD if side == 'right' else BG_EQ
                fg = TX_ADD if side == 'right' else TX_GREY
            else:
                bg, fg = BG_EQ, TX_NORM

            # background rect
            rect = FancyBboxPatch(
                (0.0, i), 1.0, 1.0,
                boxstyle="square,pad=0",
                facecolor=bg, edgecolor=BORDER, linewidth=0.3,
                transform=ax.transData, zorder=0)
            ax.add_patch(rect)

            # text
            if text:
                ax.text(0.015, i + 0.5, text, fontsize=fs, color=fg,
                        family='monospace', va='center', ha='left',
                        transform=ax.transData, zorder=1)

    # ── Legend ────────────────────────────────────────────────────────────
    legend_elements = [
        Patch(facecolor=BG_DEL, edgecolor=BORDER, label='Changed / Removed'),
        Patch(facecolor=BG_ADD, edgecolor=BORDER, label='Changed / Added'),
        Patch(facecolor=BG_EQ,  edgecolor=BORDER, label='Unchanged'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=8, framealpha=0.9, edgecolor='#D1D5DB',
               bbox_to_anchor=(0.5, 0.01))

    plt.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="new_exps/e3/densesteer_good_examples_correct50.json")
    parser.add_argument("--doc_id", type=int, default=1268)
    parser.add_argument("--output", default=None)
    parser.add_argument("--index", type=int, default=None)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    if args.index is not None:
        sample = data[args.index]
    else:
        sample = next((d for d in data if d["doc_id"] == args.doc_id), None)
        if not sample:
            print(f"doc_id={args.doc_id} not found!")
            return

    out = args.output or f"new_exps/e3/diff_figure_{sample['doc_id']}.png"
    make_figure(sample["resp_before"], sample["resp_after"], sample, out)


if __name__ == "__main__":
    main()
