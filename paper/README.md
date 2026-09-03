# Paper: Composable KV

Draft paper (workshop / arXiv scale) for the composable-kv project.

## Files

- `paper.tex` — LaTeX source (single-column, `article` class)
- `references.bib` — BibTeX references (KV compression, prefix cache, representation arithmetic, interpretability)

## How to build

### Option A — Overleaf (recommended, no local install)

1. Create a new Overleaf project → Upload
2. Upload `paper.tex` and `references.bib`
3. Overleaf compiles automatically. Output is a PDF.

### Option B — Local (requires TeX Live or MiKTeX)

```bash
cd paper
pdflatex paper
bibtex paper
pdflatex paper
pdflatex paper
```

The double compile is standard for LaTeX bibliography resolution.

## Structure

- Abstract: single paragraph, includes exact held-out numbers and cost figures
- 8 sections: Introduction, Related Work, Problem Formulation, Experimental Setup, Results (layer-wise + method comparison + held-out + cost), Discussion, Limitations, Conclusion
- Appendix A: pipeline sanity checks with numerical bounds

Approximate length: 8 pages, targetable at workshops such as
NeurIPS/ICLR Efficient ML / ES-FoMo. arXiv submission (cs.LG or cs.CL)
requires an endorser for first-time submitters.

## What still needs to be added before submission

Empirical:

- Reproduce main held-out numbers on Qwen-2.5-1.5B (baseline reliability)
- Scale n from 20 to 100+, with explicit entity / length / template
  variation controlled in splits
- Optionally: single figure (main-result bar chart or Pareto plot)

Writeup:

- Replace `<repo-to-be-added>` with actual GitHub URL
- Fill in institution / affiliation if applicable (currently
  "Independent research")
- Sanity check appendix could be expanded with per-example tables
  if space permits
- One canonical figure would help; PGFPlots or an imported PDF works
