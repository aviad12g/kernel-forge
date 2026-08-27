from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
OVERLEAF = ROOT / "paper" / "overleaf"
OVERLEAF_TABLES = OVERLEAF / "tables"
OVERLEAF_FIGURES = OVERLEAF / "figures"


def main() -> int:
    required = [
        TABLES / "benchmark_protocol.csv",
        TABLES / "fused8_model_comparison.csv",
        TABLES / "fused8_stable_winners.csv",
        TABLES / "kernelbench_l1_pilot.csv",
        TABLES / "uncertainty_extraction_status.csv",
        TABLES / "verification_rate_intervals.csv",
        TABLES / "kernelbench_family_summary.csv",
        TABLES / "single_run_repeat_flip_summary.csv",
        TABLES / "fused8_uncertainty_recovered.csv",
        TABLES / "kernelbench_memory_filter_summary.csv",
        TABLES / "statistical_summary.csv",
        TABLES / "kernelbench_family_outcomes_detailed.csv",
        TABLES / "kernelbench_repairability_criteria.csv",
        TABLES / "kernelbench_loss_win_interpretation.csv",
        TABLES / "kernelbench_loss_mechanism_summary.csv",
        TABLES / "kernelbench_eager_baseline_notes.csv",
        TABLES / "fused8_eager_baseline_notes.csv",
        TABLES / "compile_time_summary.csv",
        TABLES / "positioning_table.csv",
        TABLES / "headline_clock_validation.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("Missing required table sources:")
        for path in missing:
            print(f"- {path}")
        return 1

    OVERLEAF_TABLES.mkdir(parents=True, exist_ok=True)
    OVERLEAF_FIGURES.mkdir(parents=True, exist_ok=True)
    _write_tables()
    _build_figures()
    _copy_references()
    _try_build_fallback_pdf()
    print("Paper assets rebuilt.")
    return 0


def _read_csv(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _speed(value: str) -> str:
    value = value.strip()
    if not value or value in {"legacy", "not preserved", "not applicable"}:
        return _latex_escape(value)
    suffix = "x" if value.replace(".", "", 1).isdigit() else ""
    if value.endswith("x"):
        suffix = ""
    return rf"{_latex_escape(value)}{suffix}"


def _write_tables() -> None:
    _write_protocol_table()
    _write_fused8_summary()
    _write_kernelbench_summary()
    _write_fused8_appendix()
    _write_kernelbench_task_appendix()
    _write_kernelbench_candidate_appendix()
    _write_repair_subset_appendix()
    _write_statistical_summary()
    _write_kernelbench_family_summary()
    _write_fused8_uncertainty_recovery()
    _write_repairability_criteria()
    _write_loss_win_interpretation()
    _write_loss_profiler_attribution()
    _write_headline_clock_validation()
    _write_eager_baseline_notes()
    _write_fused8_eager_baseline_notes()
    _write_compile_time_summary()
    _write_memory_filter_summary()
    _write_positioning_table()


def _write_protocol_table() -> None:
    rows = _read_csv("benchmark_protocol.csv")
    body = []
    for row in rows:
        samples = _latex_escape(
            f"{row.get('repeats') or ''} x {row.get('sessions') or ''}"
        )
        is_fused8 = row["study"].startswith("fused8")
        body.append(
            " & ".join(
                [
                    _latex_escape("fused8" if is_fused8 else "Historical KB"),
                    _latex_escape(row["timing"]),
                    samples,
                    "128 MB",
                    "compile",
                    _latex_escape(
                        "template; Gemini; OpenAI mini"
                        if is_fused8
                        else "affected artifacts"
                    ),
                ]
            )
            + r" \\"
        )
    _write(
        OVERLEAF_TABLES / "protocol_table.tex",
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{Measurement protocol. The KernelBench row describes the affected historical run and is retained only for evaluator auditing.}",
                r"\label{tab:protocol}",
                r"\small",
                r"\begin{tabularx}{\linewidth}{l l c c l X}",
                r"\toprule",
                r"Study & Timing & Samples $\times$ sessions & Cache & Baseline & Source \\",
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabularx}",
                r"\end{table}",
                "",
            ]
        ),
    )


def _write_fused8_summary() -> None:
    rows = _read_csv("fused8_model_comparison.csv")[:3]
    body = []
    for row in rows:
        body.append(
            " & ".join(
                [
                    _latex_escape(row["baseline"]),
                    row["candidates"],
                    _latex_escape(row["verified"]),
                    _speed(row["median_speedup_vs_eager"]),
                    _speed(row["median_speedup_vs_compile"]),
                    _latex_escape(row["repeat_stable_wins"]),
                ]
            )
            + r" \\"
        )
    _write(
        OVERLEAF_TABLES / "fused8_summary_table.tex",
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{Rigorous fused8 summary. Models beat the compiler baseline at median, but all sources remain below eager at median; deterministic templates remain the strongest floor.}",
                r"\label{tab:fused8-summary}",
                r"\small",
                r"\begin{tabularx}{\linewidth}{l r l r r X}",
                r"\toprule",
                r"Source & Cand. & Verified & Eager & Compile & Stable wins \\",
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabularx}",
                r"\end{table}",
                "",
            ]
        ),
    )


def _write_kernelbench_summary() -> None:
    body = [
        r"State contract & Free \code{forward(*args)} received only runtime inputs & Require \code{ModelNew} for every official \code{Model} task \\",
        r"Reference lifecycle & Rebuilt and transferred \code{Model} inside each call & Construct persistent modules from one seeded init-input snapshot \\",
        r"Static policy & Accepted high-level Torch fallback and import-time construction & Apply reachable-call AST policy and require an actual reachable Triton launch \\",
    ]
    _write(
        OVERLEAF_TABLES / "kernelbench_summary_table.tex",
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{Post-hoc KernelBench adapter audit. The corrected implementation removes three defects before external revalidation.}",
                r"\label{tab:kernelbench-summary}",
                r"\small",
                r"\begin{tabularx}{\linewidth}{l X X}",
                r"\toprule",
                r"Audit item & Historical behavior & Corrected behavior \\",
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabularx}",
                r"\end{table}",
                "",
            ]
        ),
    )


def _write_fused8_appendix() -> None:
    rows = _read_csv("fused8_stable_winners.csv")
    body = []
    for row in rows:
        body.append(
            " & ".join(
                [
                    _latex_escape(row["task"]),
                    _latex_escape(row["stable_winner"]),
                    _latex_escape(row["source_type"]),
                    _speed(row["repeat_median_speedup_vs_eager"]),
                    _latex_escape(row["uncertainty"]),
                    _latex_escape(row["interpretation"]),
                ]
            )
            + r" \\"
        )
    _write(
        OVERLEAF_TABLES / "fused8_task_appendix_table.tex",
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{Fused8 task-level repeatability summary. Imported fused8 summaries preserve medians and labels, but not all interval statistics.}",
                r"\label{tab:fused8-task-appendix}",
                r"\scriptsize",
                r"\begin{tabularx}{\linewidth}{l l l r l X}",
                r"\toprule",
                r"Task & Winner & Source & Repeat median & Uncertainty & Interpretation \\",
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabularx}",
                r"\end{table}",
                "",
            ]
        ),
    )


def _write_kernelbench_task_appendix() -> None:
    rows = [
        ("conv2d_square", "conv"), ("matmul_3d", "matmul"), ("maxpool1d", "pool"),
        ("conv3d_square", "conv"), ("matmul_4d", "matmul"), ("maxpool2d", "pool"),
        ("conv2d_asym_input", "conv"), ("CE", "loss"), ("matmul_diagonal", "matmul"),
        ("conv2d_asym_kernel", "conv"), ("matmul_symmetric", "matmul"), ("avgpool1d", "pool"),
        ("conv_transpose2d", "conv"), ("kldiv", "loss"), ("matmul_upper_tri", "matmul"),
        ("conv_transpose3d", "conv"), ("triplet", "loss"), ("matmul_lower_tri", "matmul"),
        ("conv3d_asym_input", "conv"), ("matmul_transposed_A", "matmul"),
    ]
    body = []
    for left, right in zip(rows[::2], rows[1::2]):
        body.append(f"{_latex_escape(left[0])} & {_latex_escape(left[1])} & {_latex_escape(right[0])} & {_latex_escape(right[1])} " + r"\\")
    _write(
        OVERLEAF_TABLES / "kernelbench_task_appendix_table.tex",
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{Task aliases from the affected historical KernelBench adapter run. The subset and its outputs are retained for auditability, not external performance inference.}",
                r"\label{tab:kernelbench-task-appendix}",
                r"\scriptsize",
                r"\begin{tabularx}{\linewidth}{l l l l}",
                r"\toprule",
                r"Task alias & Family & Task alias & Family \\",
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabularx}",
                r"\end{table}",
                "",
            ]
        ),
    )


def _write_kernelbench_candidate_appendix() -> None:
    rows = [
        ("CE", "loss", "stable", "1.992x", "IQR 0.0022"),
        ("triplet", "loss", "stable", "4.176x", "IQR 0.0005"),
        ("matmul_diagonal", "matmul", "below", "0.984x", "IQR 0.0016"),
        ("numeric failures", "mixed", "failed", "9 tasks", "dominant failure category"),
        ("timeout/OOM", "mixed", "failed", "4 tasks", "candidate execution failures"),
        ("runtime", "conv", "failed", "2 tasks", "implementation exceptions"),
        ("Triton compile", "mixed", "failed", "2 tasks", "compile errors"),
    ]
    body = [" & ".join(map(_latex_escape, row)) + r" \\" for row in rows]
    _write(
        OVERLEAF_TABLES / "kernelbench_candidate_appendix_table.tex",
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{Outcomes recorded by the affected historical KernelBench evaluator. Current policy and state-contract corrections prevent interpreting these rows as model capability.}",
                r"\label{tab:kernelbench-candidate-appendix}",
                r"\scriptsize",
                r"\begin{tabularx}{\linewidth}{l l l l X}",
                r"\toprule",
                r"Task/category & Family & Label & Result & Note \\",
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabularx}",
                r"\end{table}",
                "",
            ]
        ),
    )


def _write_repair_subset_appendix() -> None:
    rows = [
        ("matmul_3d", "numeric", "high", "failed"),
        ("matmul_symmetric", "numeric", "high", "failed"),
        ("matmul_upper_tri", "numeric", "high", "failed"),
        ("matmul_lower_tri", "numeric", "high", "failed"),
        ("matmul_transposed_A", "numeric", "high", "failed"),
        ("kldiv", "numeric", "high", "stable, 1.843x"),
        ("matmul_4d", "Triton compile", "high", "failed"),
        ("avgpool1d", "Triton compile", "high", "failed"),
    ]
    body = [" & ".join(map(_latex_escape, row)) + r" \\" for row in rows]
    _write(
        OVERLEAF_TABLES / "repair_subset_appendix_table.tex",
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{Repair outcomes recorded by the affected historical evaluator. The rows characterize its failure workflow and are not corrected-adapter results.}",
                r"\label{tab:repair-subset-appendix}",
                r"\scriptsize",
                r"\begin{tabularx}{\linewidth}{l l l X}",
                r"\toprule",
                r"Task & Original failure & Repairability & Repair result \\",
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabularx}",
                r"\end{table}",
                "",
            ]
        ),
    )


def _write_statistical_summary() -> None:
    interval_rows = _read_csv("verification_rate_intervals.csv")
    flip_rows = _read_csv("single_run_repeat_flip_summary.csv")
    interval_body = []
    for row in interval_rows:
        interval = f"[{row['wilson_lo']}, {row['wilson_hi']}]"
        interval_body.append(
            " & ".join(
                [
                    _latex_escape(row["study"]),
                    _latex_escape(row["source"]),
                    f"{row['successes']}/{row['trials']}",
                    _latex_escape(row["rate"]),
                    _latex_escape(interval),
                ]
            )
            + r" \\"
        )
    flip = next(
        (row for row in flip_rows if row["scope"] == "fused8 template task-best summary"),
        {},
    )
    all_candidate_flip = next(
        (row for row in flip_rows if row["scope"] == "fused8 template all 160 candidates"),
        {},
    )
    notes = [
        rf"\multicolumn{{5}}{{p{{0.92\linewidth}}}}{{Task-best template flip summary: {_latex_escape(str(flip.get('repeat_below_eager', 'not preserved')))} of {_latex_escape(str(flip.get('single_run_above_eager', 'not preserved')))} single-run-above-eager task-best rows fall below eager after repeat measurement. Full 160-candidate flip frequency is {_latex_escape(str(all_candidate_flip.get('flip_rate', 'not preserved')))}.}} \\",
    ]
    _write(
        OVERLEAF_TABLES / "statistical_summary_table.tex",
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{Statistical appendix summary. Fused8 intervals describe supported fixed-budget observations; KernelBench intervals describe affected historical evaluator output only.}",
                r"\label{tab:statistical-summary}",
                r"\scriptsize",
                r"\begin{tabularx}{\linewidth}{l l c c l}",
                r"\toprule",
                r"Study & Source & Verified & Rate & Wilson 95\% CI \\",
                r"\midrule",
                *interval_body,
                r"\midrule",
                *notes,
                r"\bottomrule",
                r"\end{tabularx}",
                r"\end{table}",
                "",
            ]
        ),
    )


def _write_kernelbench_family_summary() -> None:
    detailed = TABLES / "kernelbench_family_outcomes_detailed.csv"
    family_rows = _read_csv("kernelbench_family_outcomes_detailed.csv") if detailed.exists() else _read_csv("kernelbench_family_summary.csv")
    memory_rows = _read_csv("kernelbench_memory_filter_summary.csv")
    body = []
    for row in family_rows:
        selected = row.get("selected_tasks") or row.get("selected", "")
        one_verified = row.get("one_shot_verified") or row.get("one_shot_ok", "")
        one_stable = row.get("one_shot_stable") or row.get("one_shot_stable_wins", "")
        repair_attempted = row.get("repair_attempted", "")
        repair_verified = row.get("repair_verified", "")
        combined_correct = row.get("combined_correct") or row.get("combined_unique_correct", "")
        combined_stable = row.get("combined_stable") or row.get("combined_stable_wins", "")
        body.append(
            " & ".join(
                [
                    _latex_escape(row["family"]),
                    str(selected),
                    str(one_verified),
                    str(one_stable),
                    str(repair_attempted),
                    str(repair_verified),
                    str(combined_correct),
                    str(combined_stable),
                ]
            )
            + r" \\"
        )
    skipped_by_family = [
        row for row in memory_rows if row["category"] == "skipped_memory_cap" and row["family"] != "all" and row["count"] != "0"
    ]
    memory_note = "; ".join(f"{row['family']} {row['count']}" for row in skipped_by_family)
    _write(
        OVERLEAF_TABLES / "kernelbench_family_summary_table.tex",
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{Family-level output from the affected historical KernelBench evaluator. The apparent loss-family concentration is diagnostic and not a corrected-adapter accuracy estimate.}",
                r"\label{tab:kernelbench-family-summary}",
                r"\scriptsize",
                r"\begin{tabularx}{\linewidth}{l r r r r r r r}",
                r"\toprule",
                r"Family & Sel. & 1-shot ok & 1-shot stable & Repair tried & Repair ok & Combined ok & Combined stable \\",
                r"\midrule",
                *body,
                r"\midrule",
                rf"\multicolumn{{8}}{{p{{0.92\linewidth}}}}{{Memory-cap skips encountered before the 20-task cap: {_latex_escape(memory_note)}. The selector takes the first feasible tasks in deterministic family-round-robin order; it is not random and does not scan the remaining pool after reaching the cap.}} \\",
                r"\bottomrule",
                r"\end{tabularx}",
                r"\end{table}",
                "",
            ]
        ),
    )


def _write_fused8_uncertainty_recovery() -> None:
    rows = _read_csv("fused8_uncertainty_recovered.csv")
    body = []
    for row in rows:
        body.append(
            " & ".join(
                [
                    _latex_escape(row["source"]),
                    _latex_escape(row["run_id"]),
                    _latex_escape(row["p25_p75_iqr"]),
                    _latex_escape(row["bootstrap_ci"]),
                    _latex_escape(row["per_candidate_flip_pairs"]),
                    _latex_escape(row["notes"]),
                ]
            )
            + r" \\"
        )
    _write(
        OVERLEAF_TABLES / "fused8_uncertainty_recovered_table.tex",
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{Fused8 uncertainty recovery status. The local package preserves medians and labels, but the full rigorous fused8 run directories are still required for interval and all-candidate flip recovery.}",
                r"\label{tab:fused8-uncertainty-recovery}",
                r"\scriptsize",
                r"\begin{tabularx}{\linewidth}{l l l l l X}",
                r"\toprule",
                r"Source & Run & IQR & Bootstrap CI & Flip pairs & Notes \\",
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabularx}",
                r"\end{table}",
                "",
            ]
        ),
    )


def _write_repairability_criteria() -> None:
    rows = _read_csv("kernelbench_repairability_criteria.csv")
    body = []
    for row in rows:
        body.append(
            " & ".join(
                [
                    _latex_escape(row["repairability"]),
                    _latex_escape(row["operational_criterion"]),
                    _latex_escape(row["selected_count"]),
                    _latex_escape(row["observed_total_count"]),
                    _latex_escape(row["notes"]),
                ]
            )
            + r" \\"
        )
    _write(
        OVERLEAF_TABLES / "kernelbench_repairability_criteria_table.tex",
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{Repairability categories used by the affected historical evaluator. They document selection provenance but do not validate the resulting repair rows.}",
                r"\label{tab:kernelbench-repairability}",
                r"\scriptsize",
                r"\begin{tabularx}{\linewidth}{l X r r X}",
                r"\toprule",
                r"Label & Operational criterion & Selected & Observed & Notes \\",
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabularx}",
                r"\end{table}",
                "",
            ]
        ),
    )


def _write_loss_win_interpretation() -> None:
    rows = _read_csv("kernelbench_loss_win_interpretation.csv")
    body = []
    for row in rows:
        body.append(
            " & ".join(
                [
                    _latex_escape(row["task"].replace("CrossEntropyLoss", "CE").replace("TripletMarginLoss", "Triplet").replace("KLDivLoss", "KLDiv")),
                    _latex_escape(row["speedup_vs_eager"]),
                    _latex_escape(row["speedup_vs_compile"]),
                    _latex_escape(row["likely_mechanism"]),
                    _latex_escape(row["caveat"]),
                ]
            )
            + r" \\"
        )
    _write(
        OVERLEAF_TABLES / "kernelbench_loss_win_interpretation_table.tex",
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{Historical loss-candidate source notes. The affected reference lifecycle prevents causal or performance attribution from these rows.}",
                r"\label{tab:kernelbench-loss-interpretation}",
                r"\scriptsize",
                r"\begin{tabularx}{\linewidth}{l r r X X}",
                r"\toprule",
                r"Task & Eager & Compile & Likely mechanism & Caveat \\",
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabularx}",
                r"\end{table}",
                "",
            ]
        ),
    )


def _write_loss_profiler_attribution() -> None:
    rows = _read_csv("kernelbench_loss_mechanism_summary.csv")
    op_rows = _read_csv("kernelbench_loss_profiler_ops.csv")
    memory_rows = _read_csv("kernelbench_loss_profiler_memory.csv")

    def alias(task: str) -> str:
        return (
            task.replace("CrossEntropyLoss", "CE")
            .replace("TripletMarginLoss", "Triplet")
            .replace("KLDivLoss", "KLDiv")
        )

    def calls(task: str, path: str, names: set[str]) -> int:
        total = 0
        for op in op_rows:
            if op.get("task") != task or op.get("path") != path:
                continue
            if op.get("operator") in names:
                try:
                    total += int(op.get("calls", "0"))
                except ValueError:
                    pass
        return total

    def top_ops(task: str, path: str) -> str:
        skip = {
            "Activity Buffer Request",
            "cudaDeviceSynchronize",
            "cudaLaunchKernel",
            "cuLaunchKernelEx",
            "cudaMemsetAsync",
            "aten::as_strided",
        }
        ops = []
        for op in op_rows:
            if op.get("task") != task or op.get("path") != path or op.get("status") != "profiled":
                continue
            name = op.get("operator", "")
            if not name or name in skip or name.startswith("void at::native::"):
                continue
            ops.append(name.replace("aten::", ""))
        return ", ".join(ops[:3]) if ops else "not preserved"

    def mem_gib(task: str, path: str) -> str:
        for row in memory_rows:
            if row.get("task") == task and row.get("path") == path and row.get("status") == "profiled":
                try:
                    return f"{int(row['cuda_memory_bytes']) / (1024 ** 3):.2f} GiB"
                except (KeyError, ValueError):
                    return "not preserved"
        return "not preserved"

    body = []
    for row in rows:
        task = row["task"]
        eager_launches = calls(task, "eager", {"cudaLaunchKernel", "cuLaunchKernelEx"})
        cand_launches = calls(task, "candidate", {"cudaLaunchKernel", "cuLaunchKernelEx"})
        eager_summary = f"{top_ops(task, 'eager')}; {eager_launches} launches; peak {mem_gib(task, 'eager')}"
        candidate_summary = f"{top_ops(task, 'candidate')}; {cand_launches} launches; peak {mem_gib(task, 'candidate')}"
        body.append(
            " & ".join(
                [
                    _latex_escape(alias(task)),
                    _latex_escape(row["existing_speedup_vs_eager"]),
                    _latex_escape(eager_summary),
                    _latex_escape(candidate_summary),
                    _latex_escape(row.get("caveat", "diagnostic only")),
                ]
            )
            + r" \\"
        )
    _write(
        OVERLEAF_TABLES / "kernelbench_loss_profiler_attribution_table.tex",
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{Profiler output retained from historical KernelBench candidates. It inherits the affected reference lifecycle and is a debugging artifact, not mechanism evidence for a performance claim.}",
                r"\label{tab:kernelbench-loss-profiler}",
                r"\scriptsize",
                r"\begin{tabularx}{\linewidth}{l r X X X}",
                r"\toprule",
                r"Task & Speedup & Eager profile & Candidate profile & Caveat \\",
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabularx}",
                r"\end{table}",
                "",
            ]
        ),
    )


def _write_headline_clock_validation() -> None:
    rows = _read_csv("headline_clock_validation.csv")
    body = []
    for row in rows:
        body.append(
            " & ".join(
                [
                    _latex_escape(row["task"].replace("CrossEntropyLoss", "CE").replace("TripletMarginLoss", "Triplet").replace("KLDivLoss", "KLDiv")),
                    _latex_escape(row["source"]),
                    _latex_escape(row["old_label"].replace("REPEAT_STABLE_WIN", "stable").replace("SINGLE_RUN_ONLY_WIN", "single-only")),
                    _latex_escape(row["validation_label"].replace("REPEAT_STABLE_WIN", "stable").replace("INSUFFICIENT_DATA", "unavailable")),
                    _latex_escape(row["validation_speedup_median"]),
                    _latex_escape(row["label_changed"]),
                    _latex_escape(row["status"]),
                ]
            )
            + r" \\"
        )
    _write(
        OVERLEAF_TABLES / "headline_clock_validation_table.tex",
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{Historical clock-recorded rerun. Repetition under recorded clocks does not repair the affected KernelBench state contract or reference lifecycle, so these rows are debugging artifacts only.}",
                r"\label{tab:headline-clock-validation}",
                r"\scriptsize",
                r"\begin{tabularx}{\linewidth}{l l l l r l l}",
                r"\toprule",
                r"Task & Source & Old label & Validation & Speedup & Changed & Status \\",
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabularx}",
                r"\end{table}",
                "",
            ]
        ),
    )


def _write_eager_baseline_notes() -> None:
    rows = _read_csv("kernelbench_eager_baseline_notes.csv")
    body = []
    for row in rows:
        body.append(
            " & ".join(
                [
                    _latex_escape(row["task_or_family"]),
                    _latex_escape(row["likely_eager_path"]),
                    _latex_escape(row["confidence"]),
                    _latex_escape(row["caveat"]),
                ]
            )
            + r" \\"
        )
    _write(
        OVERLEAF_TABLES / "kernelbench_eager_baseline_notes_table.tex",
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{Historical KernelBench eager-path notes. They are qualitative and do not rehabilitate the affected timing comparisons.}",
                r"\label{tab:kernelbench-eager-notes}",
                r"\scriptsize",
                r"\begin{tabularx}{\linewidth}{l X l X}",
                r"\toprule",
                r"Family & Likely eager path & Conf. & Caveat \\",
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabularx}",
                r"\end{table}",
                "",
            ]
        ),
    )


def _write_fused8_eager_baseline_notes() -> None:
    rows = _read_csv("fused8_eager_baseline_notes.csv")
    body = []
    for row in rows:
        body.append(
            " & ".join(
                [
                    _latex_escape(row["task_or_family"]),
                    _latex_escape(row["likely_eager_path"]),
                    _latex_escape(row["confidence"]),
                    _latex_escape(row["caveat"]),
                ]
            )
            + r" \\"
        )
    _write(
        OVERLEAF_TABLES / "fused8_eager_baseline_notes_table.tex",
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{Qualitative fused8 eager-baseline notes. These notes are based on reference-source inspection rather than profiler traces.}",
                r"\label{tab:fused8-eager-notes}",
                r"\scriptsize",
                r"\begin{tabularx}{\linewidth}{l X l X}",
                r"\toprule",
                r"Task/family & Likely eager path & Conf. & Caveat \\",
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabularx}",
                r"\end{table}",
                "",
            ]
        ),
    )


def _write_compile_time_summary() -> None:
    rows = _read_csv("compile_time_summary.csv")
    body = []
    for row in rows:
        body.append(
            " & ".join(
                [
                    _latex_escape(row["stage"]),
                    _latex_escape(row["verified_candidates"]),
                    _latex_escape(row["compile_time_ms_available"]),
                    _latex_escape(row["compile_time_ms_summary"]),
                    _latex_escape(row["runtime_only_ms_summary"]),
                ]
            )
            + r" \\"
        )
    _write(
        OVERLEAF_TABLES / "compile_time_summary_table.tex",
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{Compile-time field availability. Historical KernelBench compile fields use obsolete accounting or are missing and are not interpreted.}",
                r"\label{tab:compile-time-summary}",
                r"\scriptsize",
                r"\begin{tabularx}{\linewidth}{l r r l l}",
                r"\toprule",
                r"Stage & Verified & Compile times & Compile summary & Runtime summary \\",
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabularx}",
                r"\end{table}",
                "",
            ]
        ),
    )


def _write_memory_filter_summary() -> None:
    rows = _read_csv("kernelbench_memory_filter_summary.csv")
    keep = [
        row
        for row in rows
        if row["category"] in {"selected_feasible", "skipped_memory_cap"} and row["family"] in {"all", "activation", "kernelbench_l1", "loss", "normalization", "pooling", "reduction", "convolution", "matmul"}
    ]
    body = []
    for row in keep:
        body.append(
            " & ".join(
                [
                    _latex_escape(row["category"].replace("_", " ")),
                    _latex_escape(row["family"]),
                    _latex_escape(row["count"]),
                    _latex_escape(row["shape_examples"]),
                ]
            )
            + r" \\"
        )
    _write(
        OVERLEAF_TABLES / "kernelbench_memory_filter_summary_table.tex",
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{Historical KernelBench memory filtering. The estimate was a lower bound and the affected run is retained only to document selection provenance.}",
                r"\label{tab:kernelbench-memory-filter}",
                r"\scriptsize",
                r"\begin{tabularx}{\linewidth}{l l r X}",
                r"\toprule",
                r"Category & Family & Count & Shape examples \\",
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabularx}",
                r"\end{table}",
                "",
            ]
        ),
    )


def _write_positioning_table() -> None:
    rows = _read_csv("positioning_table.csv")
    role_aliases = {
        "KernelBench": ("benchmark", "scale-oriented model evaluation"),
        "CUDA-L1": ("model/system", "feedback-driven CUDA optimization"),
        "Triton / GPU DSLs": ("DSL/compiler", "candidate implementation target"),
        "PyTorch 2 / TorchInductor": ("compiler baseline", "practical compile baseline"),
        "Systems measurement methodology": ("measurement", "repeatability and bias control"),
        "Verifier-guided repair / Self-Refine-style feedback": ("repair", "one capped verifier-feedback pass"),
        "OpenKernelForge": ("this paper", "evaluation protocol and artifact discipline"),
    }
    body = []
    for row in rows:
        role, relation = role_aliases.get(
            row["work_or_area"],
            (row["primary_goal"], row["relation_to_openkernelforge"]),
        )
        body.append(
            " & ".join(
                [
                    _latex_escape(row["work_or_area"]),
                    _latex_escape(role),
                    _latex_escape(relation),
                ]
            )
            + r" \\"
        )
    _write(
        OVERLEAF_TABLES / "positioning_table.tex",
        "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{Compact positioning. \okf{} contributes an evaluation protocol and artifact discipline rather than a new generator or leaderboard result.}",
                r"\label{tab:positioning}",
                r"\scriptsize",
                r"\begin{tabularx}{\linewidth}{l l X}",
                r"\toprule",
                r"Area & Main role & Relation to \okf{} \\",
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabularx}",
                r"\end{table}",
                "",
            ]
        ),
    )


def _build_figures() -> None:
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_paper_figures.py")], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _copy_references() -> None:
    source = ROOT / "paper" / "references.bib"
    target = OVERLEAF / "references.bib"
    if not source.exists():
        raise SystemExit("paper/references.bib is missing")
    shutil.copyfile(source, target)


def _try_build_fallback_pdf() -> None:
    builder = ROOT / "scripts" / "build_paper_pdf.py"
    if builder.exists():
        subprocess.run([sys.executable, str(builder)], cwd=ROOT, check=False)


if __name__ == "__main__":
    raise SystemExit(main())
