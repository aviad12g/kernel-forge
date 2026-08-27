from scripts.analyze_same_gpu_rq2_control import compare_rows


def test_compare_rows_matches_candidate_budgets() -> None:
    original = [
        {
            "candidate_budget": "1",
            "apparent_win_rate": "1.0",
            "confirmed_win_rate": "1.0",
        }
    ]
    same_gpu = [
        {
            "candidate_budget": "1",
            "apparent_win_rate": "0.75",
            "confirmed_win_rate": "0.50",
            "median_selection_optimism_log": "0.01",
            "eligible_tasks": "4",
        }
    ]

    rows = compare_rows(original, same_gpu)

    assert rows[0]["same_gpu"] == "RTX A4500"
    assert rows[0]["interpretation"] == "screening_exceeds_confirmation"
