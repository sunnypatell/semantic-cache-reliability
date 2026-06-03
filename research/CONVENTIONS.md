# Conventions: the diamond standard

The execution-facing source of truth for this paper. Every rule here was argued to ground
by adversarial review against primary sources (CTAN manuals, arXiv help, named writing and
statistics authorities). The companion file `DECISIONS.md` records what was debated and why
each choice won. When a rule and a habit disagree, the rule wins.

---

## 1. Typesetting (LaTeX)

**Engine + class.** `pdflatex` via `latexmk` (Git for Windows supplies the Perl that latexmk
needs); `\documentclass[11pt,letterpaper]{article}`, single-column, one-side.

**Fonts.** XCharter text with matched Charter mathematics, loaded as one math package:

```latex
\usepackage{amsmath}\usepackage{mathtools}      % before newtxmath
\usepackage[T1]{fontenc}
\usepackage{XCharter}
\usepackage[charter]{newtxmath}                  % do NOT also load mathdesign or amssymb
```

newtxmath provides the AMS symbol set; loading `amssymb` alongside it clashes on `\Bbbk`.
The deprecated `cal=` option is gone in current newtxmath.

**Tables.** `booktabs` (no vertical rules; `\toprule`/`\midrule`/`\bottomrule`,
`\addlinespace` between logical blocks) + `siunitx` `S` columns for decimal alignment +
`threeparttable` for notes. Never `\resizebox` a table. Bold the winning number.

**Figures.** Float `[t]`, `\centering`, `\includegraphics[width=...]` of vector PDF.
`caption` with `labelsep=endash`; `subcaption` for panels. Match figure type to body type
(see Section 2).

**References + links.** Load `hyperref` bare, then `\hypersetup{hidelinks, breaklinks, pdf
metadata}`, then `cleveref` (after hyperref). Use `\Cref`/`\cref`; never hand-type "Figure".
Non-breaking `~` before every `\cite` and `\ref`.

**Bibliography.** `biblatex` + `biber`, `style=numeric-comp`, `sorting=nyt`, `maxbibnames=99`,
`giveninits=true`, `doi=true`, `url=false`, `eprint=true`. Ship a clean `.bbl` to arXiv.

**Polish.** `microtype` (protrusion + expansion). `csquotes` `\enquote{}`. `enumitem` for
list spacing. Widow/club penalties at 10000. Title-case section headings. Do not redefine
`\Pr`. No `\today`.

**The 10 differentiators** that read as best-in-class: Charter (not Computer Modern or
Times); en-dash captions; figure font equal to body font; golden-ratio figure sizes;
embedded TrueType (`pdf.fonttype=42`); colorblind palette by default; inward ticks; grids
only when they earn it; `siunitx` for every quantity; optical micro-details (en-dash for
ranges, thin space before units, no orphaned section numbers).

Verify before submit: `latexmk` exits clean with no overfull-box or undefined-reference
warnings; `pdffonts main.pdf` shows every font embedded.

---

## 2. Figures (matplotlib)

Use `paper/paper_style.mplstyle`. It sets `text.usetex=True` with the XCharter preamble, so
figure text is literally the same typeface as the body. `usetex` calls `latex`+`dvipng`
directly and needs no Perl. Single-column width 3.3in, golden-ratio height, Okabe-Ito
colorblind-safe cycle, viridis/cividis for continuous fields, embedded fonts, vector PDF.

**The 28-point figure rubric (pass >= 26):** legible at print size; serif font matching the
body; clean math glyphs; no missing glyphs; legend size matches axes; line widths 0.5-1.5pt;
crisp markers; no aliasing; colorblind-safe and distinct; dashed styles clear; thin crisp
spines; inward consistent ticks; tick labels not overlapping; axis labels with units, not
clipped; subtle or absent grid; figure size and aspect right; balanced whitespace; legend
not occluding data; title clear; nothing cut at edges; consistent color semantics across
sibling figures; high-contrast; perceptually uniform colormap; grayscale-distinct if needed;
PDF crisp at 200% zoom; text selectable (vector); file < 5 MB; no embedded raster.

**Self-verification loop (mandatory, no human):**
1. Render the figure to a 300-DPI PNG as well as the final PDF.
2. Open the PNG as an image and look at it as a reader would, not by reading the code.
3. Score it against the 28-point rubric; list concrete visual defects (overlap, clipped
   legend, illegible text, color clash, awkward aspect ratio, drift from sibling figures).
4. Fix the plotting code, re-render, repeat until it passes.

"Looks good" is the rubric, never an assumption from the source.

---

## 3. Prose (doctorate-level, effortlessly readable)

Clarity and elegance are the same target reached from two sides: precise, well-structured
prose reads as both lucid and sophisticated. Principles, each from a named authority:

- **Reader-expectation structure** (Gopen & Swan, *The Science of Scientific Writing*): the
  reader builds meaning from position. Put old/familiar material in the **topic position**
  (sentence opening) and new/important material in the **stress position** (sentence end).
- **Subject-verb proximity** (Gopen & Swan): keep the grammatical subject early and its verb
  close; long gaps read as interruptions.
- **Characters as subjects, actions as verbs** (Williams, *Style*): name the agent, use a
  strong finite verb, and kill nominalizations ("the implementation of X improved Y" ->
  "X improved Y").
- **Cohesion** (Williams): one sentence's stress becomes the next sentence's topic; keep a
  consistent topic string through a paragraph.
- **Concision** (Zinsser; Strunk & White): cut expletives ("there is", "it is"),
  metadiscourse ("it is important to note"), redundant pairs, empty intensifiers.
- **Rhythm** (Pinker, *The Sense of Style*): vary sentence length; a short sentence after two
  long ones lands a claim. Read aloud; fix what makes you stumble.
- **Precise diction over ornament**: erudition comes from naming the exact mechanism, not
  from fancy words. A technical term earns its place only when a plain word would lose
  meaning.

**Banned (slop and ornament alike), enforced by grep before submission:** delve, leverage,
moreover/furthermore as a crutch, "it is important to note", crucial/vital/robust as filler,
illuminate, pivotal, realm, paradigm (when "model"/"assumption" is meant), unpack, embody,
tapestry, landscape (figurative), "navigating the", seamless, "a myriad of", game-changer,
"harness the power", "not only X but also Y", "Firstly/Secondly", "In conclusion", em dashes
and double dashes.

**Per-sentence check:** subject and verb close and early? agent as subject, action as verb,
no nominalization? old before new? every word earning its place? diction precise, no slop?
length varied from the neighbor?

**Per-paragraph check:** one main point, statable in a sentence? consistent topic string?
the most important claim in the final stress position?

**Final pass:** read aloud; banned-list grep returns empty.

---

## 4. Statistical rigor

- No single-run point numbers. Report mean with a 95% **BCA bootstrap** CI (10k resamples).
- Differences between methods on a shared set: paired bootstrap or McNemar; never compare
  overlapping CIs by eye.
- Many comparisons (models x domains x thresholds): correct with Holm or Benjamini-Hochberg.
- Report effect sizes (Cohen's h/d), not just p-values.
- Under class imbalance (false hits are rare), the headline metric is **PR-AUC**, with
  ROC-AUC secondary, plus precision at fixed recall as the operating-point view. Oversample
  so each cell has >= 100 positive events before trusting its CI.
- Fair comparison: fixed prompts/temperature/seeds; strongest available baselines (here:
  exact-hash, two-tier exact+semantic, cross-encoder rerank, vCache); decontamination.
- Pre-register the questions, primary metric, and analysis plan on OSF before the main run.
- Report compute and dollar cost. Report at least one honest negative result.
- Threats-to-validity taxonomy: construct, internal, external, conclusion; a defense for each.

Authorities: Gopen & Swan; Dror et al. (significance testing in NLP); Henderson et al.
(seeds and variance); Pineau ML reproducibility checklist; Benjamini-Hochberg (FDR).

---

## 5. Reproducibility, citations, integrity

- Clean repo: `src/`, `data/` with provenance + checksums, `experiments/` with exact
  commands, `results/`, `tests/`, README, LICENSE, CITATION.cff, SPDX headers, CHANGELOG,
  semver tags. Pinned dependencies. Fixed seeds. Zenodo DOI. ORCID linked.
- Licenses: the paper and figures retain full copyright (all rights reserved); the code is
  source-available for reproduction only, NOT a permissive OSS license; the labeled data is
  the author's derived work, released for reproduction only. We deliberately do NOT use CC BY:
  the work must not be repostable as someone else's (author's explicit decision).
- Every citation sourced from OpenAlex / Semantic Scholar / DBLP / Crossref; the DOI must
  resolve and the cited work must actually say what we claim. The 2025 ghost-citation wave
  desk-rejected real papers; we verify all references and spot-read the cited passage.
- ICMJE authorship; a CRediT statement even for a solo author.

---

## 6. Completeness (the easy-to-miss list)

Notation table and a locked terminology glossary; a Datasheet for Datasets for the labeled
cache-equivalence set (with explicit license-safety of any source logs); embedding model
cards with pinned versions; a filled reproducibility checklist; a broader-impact/ethics
section (a wrong cached answer in production is a real harm) plus responsible disclosure for
the collision-attack angle; a complete limitations section; a data/code availability
statement; accessibility (figure alt text, colorblind-safe, tagged PDF, plain-language
abstract); an S-prefixed supplementary manifest; a claims-to-evidence traceability table; a
teaser figure on page one; contribution bullets; a one-sentence thesis.

---

## 7. arXiv mechanics (independent author)

Endorsement is required for a first-time unaffiliated submitter (tightened January 2026):
line up a personal endorser active in `cs.IR`/`cs.CL`/`cs.LG`, or route through a peer-
reviewed venue (e.g. TMLR) first. Primary category `cs.IR` (cross-list `cs.LG`, `cs.AI`).
TeX submission: include the matching `.bbl`, no `\today`, relative case-correct paths,
PDF/PNG figures, TeX Live 2025, fonts embedded. License: arXiv's non-exclusive license to
distribute (copyright retained; NOT CC BY, so the manuscript cannot be reposted as another's
work); a DataCite DOI is minted automatically; link ORCID. The Comments field lists pages, figures, the code/data URLs, and
the OSF preregistration.

---

## Primary sources

Typography: CTAN manuals for newtx/XCharter, booktabs, siunitx, microtype, cleveref,
caption, biblatex; Bringhurst, *The Elements of Typographic Style*; Butterick, *Practical
Typography*. Figures: Tufte, *The Visual Display of Quantitative Information*; Cleveland &
McGill (1985); Rougier, Droettboom & Bourne, "Ten Simple Rules for Better Figures" (PLOS
Comp Biol, 2014); Okabe & Ito; Paul Tol; matplotlib docs. Prose: Gopen & Swan (*American
Scientist*, 1990); Williams, *Style*; Pinker, *The Sense of Style*; Sword, *Stylish Academic
Writing*; Zinsser, *On Writing Well*; Strunk & White; McEnerney (U. Chicago). Rigor: Dror et
al.; Henderson et al.; Pineau checklist; Benjamini & Hochberg (1995). Submission: arXiv help
pages; ACM artifact-badging; Gebru et al., "Datasheets for Datasets"; Mitchell et al.,
"Model Cards."
