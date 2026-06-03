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
    "tfidf-lexical": "TF-IDF (lexical)",
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
    fig, axes = plt.subplots(1, len(DOMAIN_ORDER), figsize=(6.9, 2.4), sharey=True)
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
    axes[-1].legend(loc="upper right", fontsize=6, handlelength=1.2, borderpad=0.3)
    fig.tight_layout()
    _save(fig, "fig_frontier")


def fig_pr_curves(reports) -> None:
    encs = _encoders_present(reports)
    fig, axes = plt.subplots(1, len(DOMAIN_ORDER), figsize=(6.9, 2.4), sharey=True)
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
    axes[-1].legend(loc="lower left", fontsize=6, handlelength=1.2, borderpad=0.3)
    fig.tight_layout()
    _save(fig, "fig_pr_curves")


def fig_score_overlap(reports) -> None:
    """The grey zone: similarity-score distributions for equivalent vs non-equivalent
    pairs, for one representative encoder across the three domains."""
    enc = "bge-large-en-v1.5" if ("mrpc", "bge-large-en-v1.5") in reports else _encoders_present(reports)[-1]
    fig, axes = plt.subplots(1, len(DOMAIN_ORDER), figsize=(6.9, 2.3), sharey=True)
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
        ax.set_title(DOMAIN_LABEL[dom])
        ax.set_xlabel("cosine similarity")
    axes[0].set_ylabel("density")
    axes[0].legend(loc="upper left", fontsize=6)
    fig.suptitle(f"{ENCODER_LABEL[enc]} score overlap", fontsize=9, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_score_overlap")


def fig_coverage_bars(reports) -> None:
    """Headline operating point: coverage achievable while the false-hit rate stays
    at or below 1%, per encoder and domain."""
    encs = _encoders_present(reports)
    x = np.arange(len(encs))
    width = 0.26
    fig, ax = plt.subplots(figsize=(6.9, 2.6))
    for j, dom in enumerate(DOMAIN_ORDER):
        vals = []
        for enc in encs:
            rep = reports.get((dom, enc))
            v = rep["coverage_at_max_fhr"]["0.01"]["estimate"] if rep else np.nan
            vals.append(v)
        ax.bar(x + (j - 1) * width, vals, width, label=DOMAIN_LABEL[dom],
               color=OKABE_ITO[j], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([ENCODER_LABEL[e] for e in encs], rotation=25, ha="right")
    ax.set_ylabel(r"coverage at FHR $\leq$ 1%")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    _save(fig, "fig_coverage_bars")


def fig_heatmap(reports) -> None:
    """Encoder x domain heatmap of PR-AUC (threshold-free separability)."""
    encs = _encoders_present(reports)
    M = np.full((len(encs), len(DOMAIN_ORDER)), np.nan)
    for i, enc in enumerate(encs):
        for j, dom in enumerate(DOMAIN_ORDER):
            rep = reports.get((dom, enc))
            if rep:
                M[i, j] = rep["pr_auc"]["estimate"]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    im = ax.imshow(M, cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")
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


if __name__ == "__main__":
    main()
