# OpenKernelForge Overleaf Package

This directory is self-contained for Overleaf review builds.

The checked-in `main.tex` uses a polished one-column `article` layout for
external review. It is not a substitute for a workshop's official submission
style. Before submission, apply the selected NeurIPS workshop template and
verify its page limit, anonymity policy, bibliography rules, and supplementary
material policy without changing the reported results.

## Build

1. Upload the `overleaf/` directory to Overleaf.
2. Set `main.tex` as the main file.
3. Compile with pdfLaTeX or LaTeXmk.

The project uses standard packages only: `geometry`, `microtype`, `booktabs`,
`tabularx`, `array`, `graphicx`, `caption`, `subcaption`, `xcolor`,
`hyperref`, `enumitem`, `siunitx`, `adjustbox`, `multirow`, `pdflscape`, and
`placeins`.

## Included Files

- `main.tex`
- `references.bib`
- `sections/*.tex`
- `tables/*.tex`
- `figures/*.png` and `figures/*.pdf`

No external scripts are required inside Overleaf. The tables and figures are
pre-generated from the repository CSVs and artifacts.

## Optional Local Packaging

From the repository root:

```bash
cd paper
zip -r openkernelforge_overleaf.zip overleaf
```

If compilation fails, first check that all files in this directory were
uploaded and that Overleaf is using pdfLaTeX or LaTeXmk.
