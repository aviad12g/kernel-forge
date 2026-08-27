from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "make_workshop2026_results_figure.py"
    spec = importlib.util.spec_from_file_location("make_workshop2026_results_figure", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_result_figure_builds_only_from_completed_artifact_shapes(tmp_path: Path) -> None:
    module = _load_script()
    holdout = tmp_path / "holdout.csv"
    multiplicity = tmp_path / "multiplicity.csv"
    lifecycle = tmp_path / "lifecycle.csv"
    output = tmp_path / "figure.pdf"
    holdout.write_text(
        "screening_speedup,confirmation_speedup\n1.05,1.01\n",
        encoding="utf-8",
    )
    multiplicity.write_text(
        "candidate_budget,apparent_win_rate,confirmed_win_rate\n"
        "1,0.15,0.12\n8,0.75,0.50\n",
        encoding="utf-8",
    )
    lifecycle.write_text(
        "median_host_lifecycle_inflation,median_enclosing_event_inflation\n"
        "1.5,1.01\n",
        encoding="utf-8",
    )
    module.build_figure(holdout, multiplicity, lifecycle, output)
    assert output.exists()
    assert output.stat().st_size > 0
