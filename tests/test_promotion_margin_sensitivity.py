from scripts.analyze_promotion_margin_sensitivity import summarize_rows


def test_margin_sensitivity_counts_screen_only_promotions() -> None:
    rows = [
        {"screening_speedup": "1.03", "confirmation_speedup": "1.00"},
        {"screening_speedup": "1.04", "confirmation_speedup": "1.04"},
    ]

    summaries = summarize_rows("fixture", rows, margins=(0.02, 0.05))

    assert summaries[0].screening_wins == 2
    assert summaries[0].confirmation_wins == 1
    assert summaries[0].screen_only_promotions == 1
    assert summaries[1].screening_wins == 0
    assert summaries[1].confirmation_wins == 0
