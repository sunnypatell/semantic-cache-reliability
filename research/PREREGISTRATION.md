# Preregistration: reliability of embedding-based semantic caching

**Author:** Sunny Patel (Independent Researcher) · ORCID 0009-0005-3863-7642
**Status:** registered before the main experimental run; deviations are reported in the paper.

This document fixes the questions, design, metrics, and analysis plan before the data were
analyzed, so that confirmatory claims are separable from exploratory observation.

## Research questions

- **RQ1 (false-hit economics).** How does the false-hit rate of an embedding cache vary with
  the similarity threshold, the embedding model, and the domain, and what coverage is
  achievable while holding the false-hit rate below a ceiling?
- **RQ2 (downstream cost).** How much does a served false hit degrade end-to-end task
  quality, and does the degradation depend on the cache's position in the pipeline?
- **RQ3 (calibration).** Can a threshold calibrated to the measured, asymmetric cost of a
  false hit lower expected cost relative to a fixed threshold and to a correctness-bounding
  threshold, and does such a threshold transfer across domains?

## Hypotheses

- H1: The domain explains more variance in separability than the embedding model.
- H2: On adversarial near-duplicate pairs, separability is close to chance for all encoders.
- H3: A false hit on the final answer costs more than a false hit on retrieved context,
  because a downstream reader can partially recover from a wrong premise.
- H4: A cost-aware threshold lowers expected cost over fixed and ceiling policies, with the
  gap growing in the false-hit penalty.

## Design

- **Domains (3):** Quora Question Pairs (questions), MRPC (news), PAWS (adversarial),
  2500 human-labeled pairs each, natural base rate preserved, fixed sampling seed.
- **Encoders (6 + 1 control):** all-MiniLM-L6-v2, all-mpnet-base-v2, e5-small-v2,
  e5-large-v2, bge-base-en-v1.5, bge-large-en-v1.5, plus a TF-IDF lexical control.
- **Downstream task:** extractive QA on SQuAD with a fixed local reader; fault injection at
  false-hit rates {1, 2, 5, 10}% under three injection models.

## Metrics

- **Primary:** PR-AUC (threshold-free, robust under class imbalance), with the
  false-hit/coverage frontier as the operating-point view.
- **Secondary:** ROC-AUC; precision at fixed recall; coverage at a 1% false-hit ceiling.
- **Downstream:** token F1 (and exact match) as a function of injected false-hit rate.
- **Calibration:** expected cost per query under a parameterized false-hit penalty.

## Analysis plan

- 95% confidence intervals by the bias-corrected and accelerated (BCa) bootstrap over pairs,
  10k resamples (acceleration estimated from a capped jackknife at large n).
- Cross-condition differences by paired bootstrap / McNemar; the family of comparisons across
  encoders, domains, and thresholds corrected for the false discovery rate (Benjamini-Hochberg).
- Effect sizes (Cohen's h for proportions, Cohen's d for means) reported with every test.
- All cross-encoder comparison uses threshold-free metrics, because cosine scores are not
  comparable across encoders; thresholds are calibrated per encoder.

## Sample size and stopping

2500 pairs per domain were chosen before the run to keep bootstrap interval widths small for
the operating points of interest while remaining laptop-feasible on CPU. No data-dependent
stopping rule is used; all encoders and domains are run to completion.
