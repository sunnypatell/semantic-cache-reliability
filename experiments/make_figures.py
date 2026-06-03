# SPDX-License-Identifier: LicenseRef-Proprietary
# Copyright (c) 2026 Sunny Patel. All rights reserved.
"""Generate every figure in the paper from the saved reliability reports.

Design follows research/CONVENTIONS.md: vector PDF output with fonts matched to the
paper body (XCharter via matplotlib usetex, with a graceful serif fallback if a TeX
render fails), an Okabe-Ito colorblind-safe palette, spare Tufte-leaning axes, and a
high-DPI PNG companion for each figure so the result can be inspected by eye.

Run after the reliability experiment:  python experiments/make_figures.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RELIABILITY = ROOT / "results" / "reliability"
FIGDIR = ROOT / "paper" / "figures"
STYLE = ROOT / "paper" / "paper_style.mplstyle"

# Okabe-Ito colorblind-safe palette (consistent encoder -> color across all figures).
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
ENCODER_ORDER = [
    "all-MiniLM-L6-v2", "all-mpnet-base-v2", "e5-small-v2", "e5-large-v2",
    "bge-base-en-v1.5", "bge-large-en-v1.5", "tfidf-lexical",
]
ENCODER_LABEL = {
    "all-MiniLM-L6-v2": "MiniLM-L6", "all-mpnet-base-v2": "MPNet-base",
    "e5-small-v2": "E5-small", "e5-large-v2": "E5-large",
    "bge-base-en-v1.5": "BGE-base", "bge-large-en-v1.5": "BGE-large",
    "tfidf-lexical": "TF-IDF (lexical)", "cross-encoder-stsb": "Cross-encoder",
}
DOMAIN_ORDER = ["qqp", "mrpc", "paws"]
DOMAIN_LABEL = {"qqp": "Questions (QQP)", "mrpc": "News (MRPC)", "paws": "Adversarial (PAWS)"}


def _serif_fallback() -> None:
    # A classic serif available on the system, close to the paper's body type, with no
    # LaTeX dependency. Times New Roman if present, else matplotlib's bundled DejaVu Serif.
    matplotlib.rcParams.update({
        "text.usetex": False, "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })


def _configure_style(use_tex: bool = False) -> None:
    plt.style.use(str(STYLE))
    if not use_tex:
        # Default: no LaTeX dependency at all, so figure rendering can never trigger a
        # MiKTeX on-the-fly package install (and its admin/UAC prompt).
        _serif_fallback()
        return
    try:  # opt-in: match the paper's Charter font via usetex (when packages are present)
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.set_xlabel(r"$\tau$")
        fig.canvas.draw()
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - environment dependent
        warnings.warn(f"usetex unavailable ({exc}); falling back to serif.")
        _serif_fallback()


def load_reports() -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for path in sorted(RELIABILITY.glob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        out[(obj["dataset"], obj["encoder"])] = obj
    return out


def _encoders_present(reports) -> list[str]:
    present = {e for (_, e) in reports}
    return [e for e in ENCODER_ORDER if e in present]


def _color(enc: str) -> str:
    return OKABE_ITO[ENCODER_ORDER.index(enc) % len(OKABE_ITO)]


def _save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"{name}.pdf")
    fig.savefig(FIGDIR / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"[fig] {name}.pdf + .png")


def fig_frontier(reports) -> None:
    """Teaser: the false-hit / coverage frontier, one panel per domain.

    Each curve sweeps the threshold; the y-axis is coverage (benefit) and the x-axis is
    the false-hit rate among served queries (silent cost). Up-and-to-the-left is better.
    A vertical guide marks the 1% false-hit ceiling.
    """
    encs = _encoders_present(reports)
    fig, axes = plt.subplots(1, len(DOMAIN_ORDER), figsize=(6.9, 2.7), sharey=True)
    for ax, dom in zip(axes, DOMAIN_ORDER):
        for enc in encs:
            rep = reports.get((dom, enc))
            if rep is None:
                continue
            c = rep["curves"]["fhr_coverage"]
            fhr = np.array(c["fhr"], float)
            cov = np.array(c["coverage"], float)
            good = ~np.isnan(fhr)
            ax.plot(fhr[good], cov[good], color=_color(enc), linewidth=1.1,
                    label=ENCODER_LABEL[enc])
        ax.axvline(0.01, color="0.6", linewidth=0.6, linestyle=(0, (3, 2)))
        ax.set_title(DOMAIN_LABEL[dom])
        ax.set_xlabel("false-hit rate")
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("coverage")
    # Shared legend below the panels so it never occludes the curves (the PAWS panel
    # fills its upper-right corner, where a per-panel legend would sit).
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(encs), fontsize=6.5,
               frameon=False, handlelength=1.4, columnspacing=1.0, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, "fig_frontier")


def fig_pr_curves(reports) -> None:
    encs = _encoders_present(reports)
    fig, axes = plt.subplots(1, len(DOMAIN_ORDER), figsize=(6.9, 2.7), sharey=True)
    for ax, dom in zip(axes, DOMAIN_ORDER):
        base = None
        for enc in encs:
            rep = reports.get((dom, enc))
            if rep is None:
                continue
            base = rep["base_rate"]
            pr = rep["curves"]["pr"]
            ax.plot(pr["recall"], pr["precision"], color=_color(enc), linewidth=1.0,
                    label=ENCODER_LABEL[enc])
        if base is not None:
            ax.axhline(base, color="0.6", linewidth=0.6, linestyle=(0, (3, 2)))
        ax.set_title(DOMAIN_LABEL[dom])
        ax.set_xlabel("recall")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("precision")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(encs), fontsize=6.5,
               frameon=False, handlelength=1.4, columnspacing=1.0, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, "fig_pr_curves")


def fig_score_overlap(reports) -> None:
    """The grey zone (mechanistic money shot): similarity-score distributions for equivalent
    vs non-equivalent pairs, for the strongest encoder, with the operating threshold that
    keeps 90% recall, the false-hit zone shaded, and the median gap annotated. On the
    adversarial domain the two classes collapse into the same narrow high band."""
    enc = "bge-large-en-v1.5" if ("mrpc", "bge-large-en-v1.5") in reports else _encoders_present(reports)[-1]
    fig, axes = plt.subplots(1, len(DOMAIN_ORDER), figsize=(6.9, 2.5))
    for ax, dom in zip(axes, DOMAIN_ORDER):
        rep = reports.get((dom, enc))
        if rep is None:
            continue
        pos = np.array(rep["curves"]["scores"]["pos"], float)
        neg = np.array(rep["curves"]["scores"]["neg"], float)
        lo = float(min(pos.min(), neg.min()))
        hi = float(max(pos.max(), neg.max()))
        bins = np.linspace(lo, hi, 40)
        ax.hist(neg, bins=bins, density=True, color="#D55E00", alpha=0.55, label="not equivalent")
        ax.hist(pos, bins=bins, density=True, color="#0072B2", alpha=0.55, label="equivalent")
        # Operating threshold that retains 90% of equivalent pairs (recall 0.9).
        tau = float(np.percentile(pos, 10))
        ax.axvline(tau, color="0.2", lw=0.9, ls=(0, (3, 2)))
        ax.axvspan(tau, hi + (hi - lo) * 0.03, color="#D55E00", alpha=0.10, lw=0)
        ax.annotate(r"$\tau$ at 90% recall", xy=(tau, 0.98), xycoords=("data", "axes fraction"),
                    fontsize=5.5, color="0.2", ha="center", va="top",
                    xytext=(0, -1), textcoords="offset points")
        gap = float(np.median(pos) - np.median(neg))
        ax.text(0.04, 0.96, f"median gap = {gap:.2g}", transform=ax.transAxes,
                fontsize=6.5, va="top", ha="left")
        ax.set_title(DOMAIN_LABEL[dom])
        ax.set_xlabel("cosine similarity")
        ax.set_xlim(lo, hi + (hi - lo) * 0.03)
    axes[0].set_ylabel("density")
    # Shared legend below the panels so it never collides with the dense distributions.
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#0072B2", alpha=0.55, label="equivalent"),
               Patch(facecolor="#D55E00", alpha=0.55, label="not equivalent"),
               Patch(facecolor="#D55E00", alpha=0.10, label="false-hit zone (served, wrong)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.5, frameon=False,
               title=f"strongest encoder: {ENCODER_LABEL[enc]}", title_fontsize=6.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, "fig_score_overlap")


def fig_downstream(_reports=None) -> None:
    """End-to-end task F1 vs injected false-hit rate, by fault position, with 95% bootstrap
    bands over the 400 questions. The mixture is an exact identity in expectation; the band is
    the sampling uncertainty in the two measured endpoints."""
    import pandas as pd
    perq = ROOT / "results" / "downstream_perquestion.csv"
    if not perq.exists():
        print("[fig] downstream_perquestion.csv missing; skipping fig_downstream")
        return
    d = pd.read_csv(perq)
    correct = d["f1_correct"].to_numpy()
    faults = [  # (column, label, color)
        ("f1_wrong_ans", "answer cache (wrong answer)", "#D55E00"),
        ("f1_wrong_ctx", "context cache (wrong passage)", "#0072B2"),
        ("f1_trunc", "context cache (truncated)", "#009E73"),
    ]
    n = len(d)
    ps = np.linspace(0.0, 0.10, 21)
    marks = np.array([0.0, 0.01, 0.02, 0.05, 0.10])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(2000, n))           # shared resamples across conditions
    bc = correct[idx].mean(axis=1)                      # (B,)
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for col, label, color in faults:
        fault = d[col].to_numpy()
        line = (1 - ps) * correct.mean() + ps * fault.mean()
        bf = fault[idx].mean(axis=1)                     # (B,)
        curves = (1 - ps[None, :]) * bc[:, None] + ps[None, :] * bf[:, None]
        lo, hi = np.percentile(curves, [2.5, 97.5], axis=0)
        ax.fill_between(ps * 100, lo, hi, color=color, alpha=0.16, linewidth=0)
        ax.plot(ps * 100, line, linewidth=1.2, color=color, label=label)
        mline = (1 - marks) * correct.mean() + marks * fault.mean()
        ax.plot(marks * 100, mline, "o", markersize=3, color=color)
    ax.set_xlabel("injected false-hit rate (%)")
    ax.set_ylabel(r"end-to-end token $F_1$")
    ax.set_xlim(0, 10)
    ax.legend(fontsize=6.2, loc="lower left")
    fig.tight_layout()
    _save(fig, "fig_downstream")


def fig_coverage_bars(reports) -> None:
    """Headline operating point: coverage achievable while the false-hit rate stays
    at or below 1%, per encoder and domain."""
    encs = _encoders_present(reports)
    x = np.arange(len(encs))
    width = 0.26
    fig, ax = plt.subplots(figsize=(6.9, 2.6))
    for j, dom in enumerate(DOMAIN_ORDER):
        vals, lo_err, hi_err = [], [], []
        for enc in encs:
            rep = reports.get((dom, enc))
            if rep:
                ci = rep["coverage_at_max_fhr"]["0.01"]
                est = ci["estimate"]
                vals.append(est)
                # Asymmetric 95% BCa bootstrap interval (estimate is not the midpoint of a
                # skewed rare-event rate), clipped so a degenerate bound never draws below zero.
                lo_err.append(max(0.0, est - ci.get("low", est)))
                hi_err.append(max(0.0, ci.get("high", est) - est))
            else:
                vals.append(np.nan); lo_err.append(0.0); hi_err.append(0.0)
        ax.bar(x + (j - 1) * width, vals, width, label=DOMAIN_LABEL[dom],
               color=OKABE_ITO[j], edgecolor="black", linewidth=0.4,
               yerr=[lo_err, hi_err],
               error_kw=dict(elinewidth=0.6, capsize=1.5, capthick=0.6, ecolor="0.25"))
    ax.set_xticks(x)
    ax.set_xticklabels([ENCODER_LABEL[e] for e in encs], rotation=25, ha="right")
    ax.set_ylabel(r"coverage at FHR $\leq$ 1%")
    # Every value is below 0.10 (the point of the figure), so zoom in to make the
    # model and domain differences legible while the absolute scale stays low.
    ax.set_ylim(0, 0.20)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    _save(fig, "fig_coverage_bars")


def fig_heatmap(reports) -> None:
    """Encoder x domain heatmap of PR-AUC (threshold-free separability). The cross-encoder
    verifier is shown as a final row to make its adversarial collapse visible."""
    encs = _encoders_present(reports)
    if ("paws", "cross-encoder-stsb") in reports:
        encs = encs + ["cross-encoder-stsb"]
    M = np.full((len(encs), len(DOMAIN_ORDER)), np.nan)
    for i, enc in enumerate(encs):
        for j, dom in enumerate(DOMAIN_ORDER):
            rep = reports.get((dom, enc))
            if rep:
                M[i, j] = rep["pr_auc"]["estimate"]
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    im = ax.imshow(M, cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")
    if "cross-encoder-stsb" in encs:  # set the verifier row apart from the cache encoders
        ax.axhline(len(encs) - 1.5, color="white", linewidth=1.6)
    ax.set_xticks(range(len(DOMAIN_ORDER)))
    ax.set_xticklabels([DOMAIN_LABEL[d].split(" ")[0] for d in DOMAIN_ORDER])
    ax.set_yticks(range(len(encs)))
    ax.set_yticklabels([ENCODER_LABEL[e] for e in encs])
    for i in range(len(encs)):
        for j in range(len(DOMAIN_ORDER)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] < 0.78 else "black", fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("PR-AUC", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_heatmap")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Generate paper figures from reliability reports.")
    ap.add_argument("--usetex", action="store_true",
                    help="render text with LaTeX to match the paper's Charter font (opt-in)")
    args = ap.parse_args()
    _configure_style(use_tex=args.usetex)
    reports = load_reports()
    if not reports:
        raise SystemExit("No reliability reports found; run run_reliability.py first.")
    print(f"[figures] loaded {len(reports)} reports; encoders: {_encoders_present(reports)}")
    fig_frontier(reports)
    fig_pr_curves(reports)
    fig_score_overlap(reports)
    fig_coverage_bars(reports)
    fig_heatmap(reports)
    fig_downstream()


if __name__ == "__main__":
    main()
