from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


def test_workshop_protocol_is_preregistered_without_results() -> None:
    protocol = yaml.safe_load(
        (ROOT / "configs" / "workshop2026_holdout_protocol.yaml").read_text(encoding="utf-8")
    )
    assert protocol["study"]["status"] == "preregistered_pending_gpu_execution"
    assert protocol["candidate_generation"]["enabled"] is False
    assert protocol["kernelbench"]["target_tasks"] == 48
    amendment = protocol["study"]["protocol_amendment"]
    assert amendment["timing"] == "before_task_manifest_freeze_and_candidate_generation"
    assert amendment["change"] == "target_tasks_50_to_48"
    assert protocol["confirmation"]["fresh_processes"] == 7
    assert protocol["promotion"]["practical_speedup_margin"] == 0.02
    assert protocol["correctness"]["seeds"] == [1103, 2207, 3301, 4409, 5519]
    assert protocol["aggregate_analysis"]["primary"] is True
    assert protocol["controls"]["lifecycle"]["execution"].startswith("separate_")

    multiplicity = yaml.safe_load(
        (ROOT / "configs" / "workshop2026_multiplicity_protocol.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert multiplicity["candidates"]["variants_per_task"] == 20
    assert multiplicity["analysis"]["all_candidates_require_independent_confirmation"] is True


def test_workshop_source_uses_official_style_and_separate_main_label() -> None:
    paper = ROOT / "paper" / "workshop2026"
    assert (paper / "neurips_2026.sty").exists()
    main = (paper / "main.tex").read_text(encoding="utf-8")
    assert "\\usepackage[sglblindworkshop]{neurips_2026}" in main
    assert "Workshop review draft. Not submitted." in main
    assert "Auditing LLM-Generated GPU Kernel Claims" in main
    assert (paper / "submission.tex").read_text(encoding="utf-8").startswith(
        "\\def\\okfsubmissionupload"
    )
    assert "\\label{mainmatterend}" in main
    assert "\\bibliography{references}" in main


def test_workshop_paper_reports_completed_corrected_results() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "paper" / "workshop2026").rglob("*.tex"))
    )
    assert "measurements are pending" not in source
    assert "pendingcampaign" not in source
    assert "Every frozen winner was below eager" in source
    assert "all 80 deterministic candidates" in source
    assert "three of four task winners" in source
    assert "1.0001$\\times$" in source
    assert "0.368 worker-hours" in source
    assert "2.091 recorded worker-hours" in source
    assert "not inferred from the KernelBench pool" in source
    assert "77 failed static policy" in source
    assert "1.826$\\times$" in source
