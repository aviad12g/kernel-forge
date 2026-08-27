from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper/openkernelforge_paper.pdf"
NOTES = ROOT / "paper/pdf_build_notes.md"
OVERLEAF = ROOT / "paper/overleaf"


PAPER_FILES = [
    ROOT / "paper/paper.md",
    ROOT / "paper/methodology.md",
    ROOT / "paper/experiments.md",
    ROOT / "paper/related_work.md",
    ROOT / "paper/limitations.md",
]
APPENDIX_FILE = ROOT / "paper/kernelbench_appendix.md"

FIGURE_MAP = {
    "openkernelforge_pipeline": ROOT / "reports/figures/openkernelforge_pipeline.png",
    "fused8_stable_speedups": ROOT / "reports/figures/fused8_stable_speedups.png",
    "fused8_source_summary": ROOT / "reports/figures/fused8_source_summary.png",
    "bias_relu_single_run_flip": ROOT / "reports/figures/bias_relu_single_run_flip.png",
    "kernelbench_pilot_funnel": ROOT / "reports/figures/kernelbench_pilot_funnel.png",
    "kernelbench_failure_taxonomy": ROOT / "reports/figures/kernelbench_failure_taxonomy.png",
}


def main() -> int:
    try:
        build_pdf()
    except Exception as exc:  # pragma: no cover - exercised manually
        NOTES.write_text(
            "# PDF Build Notes\n\n"
            "PDF build failed.\n\n"
            f"Error: `{type(exc).__name__}: {exc}`\n\n"
            "Required fallback dependency: `reportlab`.\n",
            encoding="utf-8",
        )
        print(f"PDF build failed: {exc}")
        return 1
    return 0


def build_pdf() -> None:
    latex_ok, latex_note, latex_builder = _try_build_with_latex()
    if latex_ok:
        NOTES.write_text(
            _build_notes(f"{latex_builder} LaTeX", latex_note), encoding="utf-8"
        )
        print(f"Wrote {OUTPUT}")
        print(f"Wrote {NOTES}")
        return

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        rightMargin=0.72 * inch,
        leftMargin=0.72 * inch,
        topMargin=0.72 * inch,
        bottomMargin=0.72 * inch,
        title="OpenKernelForge: Repeatability-Aware Evaluation for LLM-Generated Triton Kernels",
        author="Anonymous Authors",
    )
    story: list = []
    story.extend(_title_page(styles))
    story.append(PageBreak())

    counters = {"h1": 0, "h2": 0, "title_seen": 0}
    for path in PAPER_FILES:
        story.extend(_parse_markdown(path.read_text(encoding="utf-8"), styles, counters))
        story.append(Spacer(1, 0.12 * inch))

    story.append(PageBreak())
    story.append(Paragraph("References", styles["Heading1"]))
    story.extend(_references(styles))

    story.append(PageBreak())
    story.extend(_parse_markdown(APPENDIX_FILE.read_text(encoding="utf-8"), styles, counters))

    doc.build(story, onFirstPage=_page_number, onLaterPages=_page_number)
    NOTES.write_text(_build_notes("ReportLab fallback", latex_note), encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Wrote {NOTES}")


def _try_build_with_latex() -> tuple[bool, str, str]:
    main_tex = OVERLEAF / "main.tex"
    if not main_tex.exists():
        return False, "paper/overleaf/main.tex not found", "none"

    builders: list[tuple[str, list[str]]] = []
    if latexmk := shutil.which("latexmk"):
        builders.append(
            (
                "latexmk",
                [
                    latexmk,
                    "-g",
                    "-pdf",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "main.tex",
                ],
            )
        )
    if tectonic := shutil.which("tectonic"):
        builders.append(("tectonic", [tectonic, "main.tex"]))

    attempts: list[str] = []
    output_pdf = OVERLEAF / "main.pdf"
    for builder_name, command in builders:
        output_pdf.unlink(missing_ok=True)
        result = subprocess.run(
            command,
            cwd=OVERLEAF,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode == 0 and output_pdf.exists():
            OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_pdf, OUTPUT)
            final_log = (OVERLEAF / "main.log").read_text(
                encoding="utf-8", errors="replace"
            ) if (OVERLEAF / "main.log").exists() else result.stdout
            warning_lines = [
                line
                for line in final_log.splitlines()
                if line.startswith("LaTeX Warning:")
                or line.startswith("Package Warning:")
                or "overfull \\hbox" in line.lower()
            ]
            if warning_lines:
                note = (
                    f"{builder_name} completed with warnings:\n\n```text\n"
                    + "\n".join(warning_lines[-20:])
                    + "\n```"
                )
            else:
                note = f"{builder_name} completed without reported warnings"
            return True, note, builder_name
        log_excerpt = "\n".join(result.stdout.splitlines()[-30:])
        attempts.append(
            f"{builder_name} failed with exit {result.returncode}\n\n```text\n{log_excerpt}\n```"
        )

    if not attempts:
        attempts.append("latexmk and tectonic were not found")
    return False, "\n\n".join(attempts), "none"


def _title_page(styles: dict[str, ParagraphStyle]) -> list:
    return [
        Spacer(1, 1.0 * inch),
        Paragraph("OpenKernelForge", styles["Title"]),
        Paragraph("Repeatability-Aware Evaluation for LLM-Generated Triton Kernels", styles["Subtitle"]),
        Spacer(1, 0.35 * inch),
        Paragraph("Workshop draft for external review", styles["Centered"]),
        Spacer(1, 0.35 * inch),
        Paragraph(
            "This paper reports a controlled fused8 study and a historical KernelBench evaluator audit. Corrected external CUDA validation is pending; no KernelBench performance claim is made.",
            styles["CenteredSmall"],
        ),
    ]


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle(
            "OKFTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=29,
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "Subtitle": ParagraphStyle(
            "OKFSubtitle",
            parent=base["Title"],
            fontName="Helvetica",
            fontSize=16,
            leading=20,
            alignment=TA_CENTER,
            spaceAfter=20,
        ),
        "Heading1": ParagraphStyle(
            "OKFH1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            spaceBefore=12,
            spaceAfter=7,
        ),
        "Heading2": ParagraphStyle(
            "OKFH2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=15,
            spaceBefore=10,
            spaceAfter=5,
        ),
        "BodyText": ParagraphStyle(
            "OKFBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12.2,
            alignment=TA_LEFT,
            spaceAfter=5,
        ),
        "TableCell": ParagraphStyle(
            "OKFTableCell",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=6.9,
            leading=8.2,
            alignment=TA_LEFT,
            splitLongWords=False,
            spaceAfter=0,
        ),
        "TableHead": ParagraphStyle(
            "OKFTableHead",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=6.8,
            leading=8.1,
            alignment=TA_LEFT,
            splitLongWords=False,
            spaceAfter=0,
        ),
        "Code": ParagraphStyle(
            "OKFCode",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.5,
            leading=9,
            leftIndent=8,
            rightIndent=8,
            backColor=colors.HexColor("#f5f5f5"),
        ),
        "Centered": ParagraphStyle("Centered", parent=base["BodyText"], alignment=TA_CENTER, fontSize=11),
        "CenteredSmall": ParagraphStyle("CenteredSmall", parent=base["BodyText"], alignment=TA_CENTER, fontSize=9),
        "Caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#444444"),
            spaceAfter=8,
        ),
    }
    return styles


def _parse_markdown(text: str, styles: dict[str, ParagraphStyle], counters: dict[str, int]) -> list:
    story: list = []
    lines = text.splitlines()
    paragraph: list[str] = []
    bullets: list[str] = []
    in_code = False
    code_lines: list[str] = []
    i = 0

    def flush_paragraph() -> None:
        if paragraph:
            story.append(Paragraph(_inline(" ".join(paragraph)), styles["BodyText"]))
            paragraph.clear()

    def flush_bullets() -> None:
        if bullets:
            items = [ListItem(Paragraph(_inline(item), styles["BodyText"])) for item in bullets]
            story.append(ListFlowable(items, bulletType="bullet", leftIndent=18))
            bullets.clear()

    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith("```"):
            if in_code:
                story.append(Preformatted("\n".join(code_lines), styles["Code"]))
                code_lines.clear()
                in_code = False
            else:
                flush_paragraph()
                flush_bullets()
                in_code = True
            i += 1
            continue
        if in_code:
            code_lines.append(line)
            i += 1
            continue
        if line.startswith("| ") and i + 1 < len(lines) and lines[i + 1].startswith("| ---"):
            flush_paragraph()
            flush_bullets()
            table_lines = [line, lines[i + 1].rstrip()]
            i += 2
            while i < len(lines) and lines[i].startswith("| "):
                table_lines.append(lines[i].rstrip())
                i += 1
            story.append(_markdown_table(table_lines, styles))
            story.append(Spacer(1, 0.08 * inch))
            continue
        if line.startswith("[[PAGEBREAK]]"):
            flush_paragraph()
            flush_bullets()
            story.append(PageBreak())
            i += 1
            continue
        if line.startswith("[[FIGURE:") and line.endswith("]]"):
            flush_paragraph()
            flush_bullets()
            story.append(_figure_from_marker(line, styles))
            story.append(Spacer(1, 0.10 * inch))
            i += 1
            continue
        if not line.strip():
            flush_paragraph()
            flush_bullets()
            i += 1
            continue
        if line.startswith("# "):
            flush_paragraph()
            flush_bullets()
            title = line[2:].strip()
            if counters["title_seen"] == 0 and title.startswith("OpenKernelForge:"):
                counters["title_seen"] = 1
                i += 1
                continue
            counters["h1"] += 1
            counters["h2"] = 0
            story.append(Paragraph(_inline(f"{counters['h1']}. {title}"), styles["Heading1"]))
            i += 1
            continue
        if line.startswith("## "):
            flush_paragraph()
            flush_bullets()
            if counters["h1"] == 0:
                counters["h1"] += 1
                counters["h2"] = 0
                story.append(Paragraph(_inline(f"{counters['h1']}. {line[3:].strip()}"), styles["Heading1"]))
            else:
                counters["h2"] += 1
                story.append(Paragraph(_inline(f"{counters['h1']}.{counters['h2']} {line[3:].strip()}"), styles["Heading2"]))
            i += 1
            continue
        if line.startswith("### "):
            flush_paragraph()
            flush_bullets()
            counters["h2"] += 1
            story.append(Paragraph(_inline(f"{counters['h1']}.{counters['h2']} {line[4:].strip()}"), styles["Heading2"]))
            i += 1
            continue
        if line.startswith("- "):
            flush_paragraph()
            bullets.append(line[2:].strip())
            i += 1
            continue
        if re.match(r"^\d+\. ", line):
            flush_bullets()
            paragraph.append(line)
            i += 1
            continue
        paragraph.append(line)
        i += 1
    flush_paragraph()
    flush_bullets()
    return story


def _markdown_table(lines: list[str], styles: dict[str, ParagraphStyle]) -> Table:
    rows = []
    for row_idx, line in enumerate(lines):
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        style = styles["TableHead"] if row_idx == 0 else styles["TableCell"]
        rows.append([Paragraph(_inline(_compact_label(cell)), style) for cell in cells])
    col_count = max(len(row) for row in rows)
    widths = _table_widths(rows, col_count)
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111111")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.HexColor("#111827")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.55, colors.HexColor("#6b7280")),
                ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.HexColor("#111827")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _table_widths(rows: list[list[Paragraph]], col_count: int) -> list[float]:
    total = 7.0 * inch
    if col_count == 7:
        return [0.88 * inch, 0.85 * inch, 0.95 * inch, 0.50 * inch, 1.18 * inch, 1.05 * inch, 1.59 * inch]
    if col_count == 8:
        return [1.42 * inch, 0.72 * inch, 0.48 * inch, 0.48 * inch, 0.86 * inch, 0.60 * inch, 0.66 * inch, 1.78 * inch]
    if col_count == 6:
        return [1.12 * inch, 0.85 * inch, 0.78 * inch, 0.72 * inch, 1.28 * inch, 2.25 * inch]
    if col_count == 5:
        return [1.55 * inch, 1.20 * inch, 1.05 * inch, 1.20 * inch, 2.00 * inch]
    if col_count == 4:
        return [1.30 * inch, 1.40 * inch, 1.75 * inch, 2.55 * inch]
    return [total / col_count] * col_count


def _compact_label(text: str) -> str:
    replacements = {
        "deterministic template": "template",
        "deterministic templates": "templates",
        "Gemini 3.1 Flash-Lite": "Gemini",
        "OpenAI `gpt-5.4-mini`": "OpenAI mini",
        "OpenAI gpt-5.4-mini": "OpenAI mini",
        "`torch.compile max-autotune`": "compile max-autotune",
        "torch.compile max-autotune": "compile max-autotune",
        "residual_add_relu": "residual",
        "sigmoid_mul": "sigmoid",
        "rmsnorm_small": "rmsnorm",
        "layernorm_small": "layernorm",
        "REPEAT_STABLE_WIN": "stable",
        "SINGLE_RUN_ONLY_WIN": "single-only",
        "BELOW_EAGER": "below",
        "VERIFICATION_FAILED": "failed",
        "numerical mismatch": "numeric",
        "Triton compile error": "Triton compile",
        "runtime exception": "runtime",
        "convolution": "conv",
        "Selected for repair": "Selected",
        "Original failure category": "Original fail",
        "CrossEntropyLoss": "CE",
        "TripletMarginLoss": "Triplet",
        "Candidates": "Cand.",
        "Verified": "Verif.",
        "Uncertainty": "Uncert.",
        "Closest comparison": "Closest comp.",
        "Candidate source": "Source",
        "Benchmarked": "Bench.",
        "Stable speedups": "Stable",
        "Main conclusion": "Conclusion",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    citation_labels = {
        "tillet2019triton": "Tillet et al., 2019",
        "ansel2024pytorch2": "Ansel et al., 2024",
        "kernelbench2025": "Ouyang et al., 2025",
        "dao2022flashattention": "Dao et al., 2022",
        "nvidiaCudaEvents": "NVIDIA CUDA Events",
        "cudallm2025": "Li et al., 2025",
        "ragankelley2013halide": "Ragan-Kelley et al., 2013",
        "chen2018tvm": "Chen et al., 2018",
        "madaan2023selfrefine": "Madaan et al., 2023",
    }

    def replace_citation(match: re.Match[str]) -> str:
        keys = [key.strip().lstrip("@") for key in match.group(1).split(";")]
        labels = [citation_labels.get(key, key) for key in keys]
        return "[" + "; ".join(labels) + "]"

    text = re.sub(r"\[(@[^\]]+)\]", replace_citation, text)
    return text


def _figure_from_marker(line: str, styles: dict[str, ParagraphStyle]) -> KeepTogether | Paragraph:
    content = line.removeprefix("[[FIGURE:").removesuffix("]]")
    parts = [part.strip() for part in content.split("|")]
    key = parts[0]
    caption = parts[1] if len(parts) > 1 else key.replace("_", " ")
    width = 6.4 * inch
    if len(parts) > 2:
        for part in parts[2:]:
            if part.startswith("width="):
                try:
                    width = float(part.split("=", 1)[1]) * inch
                except ValueError:
                    pass
    figure = FIGURE_MAP.get(key)
    if not figure or not figure.exists():
        return Paragraph(f"Figure missing: {key}", styles["BodyText"])
    return KeepTogether(
        [
            Image(str(figure), width=width, height=_image_height(figure, width)),
            Paragraph(_inline(caption), styles["Caption"]),
        ]
    )


def _references(styles: dict[str, ParagraphStyle]) -> list:
    refs = [
        "Tillet, P., Kung, H. T., and Cox, D. (2019). Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations. MAPL.",
        "Ansel, J. et al. (2024). PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation. ASPLOS.",
        "Ouyang, A., Guo, S., Arora, S., Zhang, A. L., Hu, W., Re, C., and Mirhoseini, A. (2025). KernelBench: Can LLMs Write Efficient GPU Kernels? arXiv:2502.10517.",
        "Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and Re, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. arXiv:2205.14135.",
        "NVIDIA. CUDA Runtime API: Event Management. Documentation for event creation, synchronization, and elapsed-time measurement.",
        "Li, X., Wang, A., Wang, G., Li, J., and Shum, C. (2025). CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning. arXiv:2507.14111.",
        "Zhang, Y., Yu, P., Wang, J., Fan, M. (X.), Reed, J., Mirhoseini, A., and Su, W. (2026). KernelBench-Verified: Do LLM-Generated Kernels Actually Beat PyTorch? arXiv:2607.16241.",
        "Ragan-Kelley, J. et al. (2013). Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines. PLDI.",
        "Chen, T. et al. (2018). TVM: An Automated End-to-End Optimizing Compiler for Deep Learning. OSDI.",
        "Madaan, A. et al. (2023). Self-Refine: Iterative Refinement with Self-Feedback. NeurIPS.",
        "Mytkowicz, T., Diwan, A., Hauswirth, M., and Sweeney, P. F. (2009). Producing Wrong Data Without Doing Anything Obviously Wrong! ASPLOS.",
        "Curtsinger, C. and Berger, E. D. (2013). STABILIZER: Statistically Sound Performance Evaluation. ASPLOS.",
        "Touati, S. (2009). Towards a Statistical Methodology to Evaluate Program Speedups and their Optimisation Techniques. arXiv:0902.1035.",
    ]
    return [Paragraph(ref, styles["BodyText"]) for ref in refs]


def _image_height(path: Path, width: float) -> float:
    try:
        from PIL import Image as PILImage  # type: ignore

        with PILImage.open(path) as img:
            w, h = img.size
            return width * h / w
    except Exception:
        return 3.5 * inch


def _page_number(canvas, doc) -> None:  # type: ignore[no-untyped-def]
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawRightString(7.8 * inch, 0.45 * inch, f"Page {doc.page}")
    canvas.restoreState()


def _build_notes(build_path: str, latex_note: str) -> str:
    pandoc = shutil.which("pandoc")
    pdflatex = shutil.which("pdflatex")
    latexmk = shutil.which("latexmk")
    tectonic = shutil.which("tectonic")
    typst = shutil.which("typst")
    quarto = shutil.which("quarto")
    return "\n".join(
        [
            "# PDF Build Notes",
            "",
            "Output: `paper/openkernelforge_paper.pdf`",
            "",
            f"Build path used: {build_path}.",
            "",
            "Preferred external builders found:",
            "",
            f"- pdflatex: `{pdflatex or 'not found'}`",
            f"- latexmk: `{latexmk or 'not found'}`",
            f"- tectonic: `{tectonic or 'not found'}`",
            f"- pandoc: `{pandoc or 'not found'}`",
            f"- typst: `{typst or 'not found'}`",
            f"- quarto: `{quarto or 'not found'}`",
            "",
            "LaTeX build note:",
            "",
            latex_note,
            "",
            "The PDF intentionally avoids phase-history language and makes no SOTA claim.",
            "",
            "Overleaf-ready source is available in `paper/overleaf/`. To build on Overleaf, upload that directory and compile `main.tex`. Locally, run `tectonic main.tex` from `paper/overleaf/`, or use `latexmk -pdf main.tex` in an environment with a full TeX Live install.",
            "",
            "The current PDF is a one-column external-review build. Apply the selected workshop's official style and page-limit rules before submission.",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
