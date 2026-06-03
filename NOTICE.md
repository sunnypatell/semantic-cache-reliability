# NOTICE: third-party data sources and attribution

The labeled cache-equivalence dataset in [`data/`](data/) is **derived** from three public,
human-annotated corpora. The author claims rights only over the **equivalence labels,
sampling, and arrangement** (see [`LICENSE`](LICENSE)). The **underlying source text is not
the author's intellectual property** and remains governed by each corpus's own license and
terms of use. Anyone reusing this data must comply with those upstream terms and provide the
attribution each requires.

| Corpus | Used as | Source | License / terms | Required attribution |
|--------|---------|--------|-----------------|----------------------|
| Quora Question Pairs (QQP) | Questions domain | Quora "First Quora Dataset Release"; accessed via the GLUE distribution (`nyu-mll/glue`, config `qqp`) | Quora dataset-release terms (restricts redistribution of the raw question text) | Quora, Inc.; cite the Quora release and GLUE (Wang et al., 2019) |
| Microsoft Research Paraphrase Corpus (MRPC) | News domain | MSR; accessed via the GLUE distribution (`nyu-mll/glue`, config `mrpc`) | Microsoft Research License Agreement (MSR-LA); restricts redistribution of the underlying news-wire text | Microsoft Research; cite Dolan & Brockett (2005) |
| PAWS (labeled_final) | Adversarial domain | Google Research (`google-research-datasets/paws`) | Released for free use; Google requests acknowledgement of Google LLC as the data source | Google LLC; cite Zhang, Baldridge & He (2019) |

> **Redistribution note.** The QQP and MRPC source corpora carry upstream terms that may
> restrict public redistribution of their raw text. See the project README and the release
> checklist for how this repository handles materialization of the labeled pairs. Verify the
> current upstream terms before any redistribution; this NOTICE is informational and is not
> legal advice.

Full bibliographic entries for all three corpora are in
[`paper/references.bib`](paper/references.bib).
