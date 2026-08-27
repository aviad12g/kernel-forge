from scripts.run_compiler_rung_confirmation import summarize_results


def test_compiler_confirmation_uses_median_across_processes() -> None:
    rows = [
        {"candidate_vs_compile_median": 1.04},
        {"candidate_vs_compile_median": 1.03},
        {"candidate_vs_compile_median": 0.99},
    ]

    summary = summarize_results(rows)

    assert summary["median_candidate_vs_compile"] == 1.03
    assert summary["above_compile_1_02"] is True
    assert summary["interpretation"] == "confirmed_above_compile_margin"
