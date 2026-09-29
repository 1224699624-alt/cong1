# AAAI 2027 Initial Draft

This directory is an isolated anonymous-submission draft based on the official AAAI 2027 Author Kit in `G:\AuthorKit27`.

Compile from this directory with:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Current evidence scope:

- Dataset: TSRS_RSNA-Epiphysis only.
- Development split: 875 train / 96 original-val.
- Compared systems: mature binary nnU-Net and mature three-class seam-supervised nnU-Net.
- clean-test-v2 is not used in this draft.
- Cross-backbone plug-in results and age/sex-conditioned priors are not yet claimed.

Static checks completed: all citation keys resolve, no bibliography entries are unused, the abstract is 185 words, figure/table references match, and no forbidden AAAI packages were added.

MiKTeX 25.12 is installed for the current Windows user. The draft has been compiled successfully into `main.pdf` with no undefined citations, undefined references, or overfull/underfull box warnings. Use `powershell -ExecutionPolicy Bypass -File build.ps1` for a complete rebuild.

Important terminology correction: under the frozen R201 code, `component_merge_rate` is the fraction of images for which the predicted binary component count is smaller than the ground-truth binary component count. It is not a pair-level adjacent-instance merge fraction.
