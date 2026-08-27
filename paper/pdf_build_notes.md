# PDF Build Notes

Output: `paper/openkernelforge_paper.pdf`

Build path used: latexmk LaTeX.

Preferred external builders found:

- pdflatex: `/Users/mazalcohen/Library/TinyTeX/bin/universal-darwin/pdflatex`
- latexmk: `/Users/mazalcohen/Library/TinyTeX/bin/universal-darwin/latexmk`
- tectonic: `/opt/homebrew/bin/tectonic`
- pandoc: `not found`
- typst: `not found`
- quarto: `not found`

LaTeX build note:

latexmk completed without reported warnings

The PDF intentionally avoids phase-history language and makes no SOTA claim.

Overleaf-ready source is available in `paper/overleaf/`. To build on Overleaf, upload that directory and compile `main.tex`. Locally, run `tectonic main.tex` from `paper/overleaf/`, or use `latexmk -pdf main.tex` in an environment with a full TeX Live install.

The current PDF is a one-column external-review build. Apply the selected workshop's official style and page-limit rules before submission.
