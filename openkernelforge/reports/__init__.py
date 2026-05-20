"""Run reporting utilities."""

from openkernelforge.reports.analyze import analyze_run
from openkernelforge.reports.compare import compare_runs_markdown
from openkernelforge.reports.gpu_debrief import debrief_gpu_run
from openkernelforge.reports.phase14 import build_phase14_report, check_research_artifacts
from openkernelforge.reports.review import review_real_run
from openkernelforge.reports.summarize import load_results, write_summary

__all__ = [
    "analyze_run",
    "build_phase14_report",
    "check_research_artifacts",
    "compare_runs_markdown",
    "debrief_gpu_run",
    "load_results",
    "review_real_run",
    "write_summary",
]
