# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Release-formality metadata: `codemeta.json`, `.zenodo.json`, `CONTRIBUTING.md`,
  `NOTICE.md` (source-corpora licenses and attribution), `data/README.md` (provenance
  datasheet), and `data/processed/SHA256SUMS` (data fixity manifest).
- Completed `CITATION.cff`: ORCID, version, release date, and a `preferred-citation`
  block so citation tools resolve to the manuscript, not just the software.
- Calibration figure in the paper body and the transfer and precision-recall figures
  in the appendix, all from the existing calibration data.
- Continuous integration: the model-free metric and statistics tests, plus a data
  integrity check, run on every push and pull request.
- Repository presentation: a hero figure in the README, an issue template, accurate
  language statistics (`.gitattributes`), and complete `pyproject` metadata.

### Changed
- LaTeX preamble hygiene: black author-note mark (`hyperfootnotes=false`), `bookmark`
  package for a robust PDF outline, `pdfsubject` metadata, and `\frenchspacing`.

### Fixed
- `fig_coverage_bars` now draws the 95% bootstrap error bars its caption describes
  (asymmetric BCa intervals from `coverage_at_max_fhr`), previously absent.
- `references.bib`: corrected the C-Pack entry to its full six-author list
  (added Defu Lian and Jian-Yun Nie), verified against arXiv 2309.07597.

## [0.1.0] - 2026-06-03

### Added
- Initial public release accompanying the manuscript "When Semantic Caches Lie".
- `cacherel` package: embeddings, cache-reliability metrics, calibration, BCa bootstrap.
- Reliability sweep over seven methods and three domains, a cross-encoder baseline, and a
  downstream fault-injection study; `experiments/run_all.py` reproduces every table and
  figure on CPU with fixed seeds.
- Labeled cache-equivalence dataset (QQP, MRPC, PAWS), 2500 pairs per domain.
- Pre-registration and a conventions knowledge base under `research/`.

[Unreleased]: https://github.com/sunnypatell/semantic-cache-reliability/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/sunnypatell/semantic-cache-reliability/releases/tag/v0.1.0
