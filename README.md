# When Semantic Caches Lie

**False hits, downstream cost, and calibration in embedding-based caching for large language models.**

An independent research project by **Sunny Patel** (Independent Researcher).

> Semantic caching reuses a stored answer whenever a new prompt looks close enough to an
> earlier one. This project asks a question the literature has mostly skipped: when the
> closeness test is wrong, what does it cost? We separate *cache-equivalence* (safe to
> reuse the stored answer) from mere *textual similarity*, measure how often embedding
> caches serve a wrong answer across models, thresholds, and domains, trace that error
> into downstream task quality, and derive a cost-aware way to set the operating point.

This repository contains the paper (LaTeX), the experiment code, the labeled dataset, and
everything needed to reproduce every number and figure.

---

## Status

Active build. The methodology, typesetting, and analysis standards are fixed; experiments
and the manuscript are in progress. Results in the paper are generated end-to-end by the
code here, with no hand-entered numbers.

## Repository layout

```
paper/            LaTeX source (main.tex), bibliography, figures, matplotlib style
src/cacherel/     Python package: embeddings, cache simulator, baselines, calibration, metrics, stats
experiments/      Runnable experiment scripts + configs
data/             Labeled cache-equivalence set (processed) + provenance; raw data is gitignored
results/          Generated tables and intermediate result files
figures/          Generated figures (vector PDF + PNG previews)
research/         Conventions knowledge base (CONVENTIONS.md) and decision record
tests/            Unit tests for metrics and statistics
```

## Reproducing the results

Requirements: Python 3.12, a TeX distribution (MiKTeX or TeX Live) with `biber`.

```powershell
# 1. Environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Build the labeled dataset, run experiments, regenerate every table and figure
python experiments\run_all.py

# 3. Compile the paper (latexmk needs Perl; Git for Windows bundles one)
cd paper
latexmk -pdf main.tex
```

Random seeds are fixed; figures and tables regenerate deterministically from `results/`.

## Licensing and intellectual property

This is original work and the author's intellectual property. Copyright (c) 2026 Sunny Patel,
all rights reserved; see `LICENSE`. The paper and figures retain full copyright (distributed
on arXiv under its non-exclusive license); the code is source-available for reproduction and
verification only; the dataset is for non-commercial research use with attribution. Any reuse,
redistribution, or derivative work requires written permission and must credit the author.
Authorship and priority are established by the public, timestamped arXiv record (DOI) and ORCID.

## Citation

See `CITATION.cff`. A DOI will be minted on release.
