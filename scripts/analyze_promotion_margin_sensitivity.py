#!/usr/bin/env python3
"""Re-evaluate frozen workshop winners across practical promotion margins."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
MARGINS = (0.0, 0.01, 0.02, 0.03, 0.05)


@dataclass(frozen=True)
class MarginSummary:
    study: str
    margin: float
    eligible_winners: int
    screening_wins: int
    confirmation_wins: int
    screen_only_promotions: int
    notes: str


def summarize_rows(
    study: str,
    rows: Iterable[dict[str, str]],
    *,
    margins: Iterable[float] = MARGINS,
    notes: str = "",
) -> list[MarginSummary]:
    pairs = [
        (float(row["screening_speedup"]), float(row["confirmation_speedup"]))
        for row in rows
        if row.get("screening_speedup") and row.get("confirmation_speedup")
    ]
    if not pairs:
        raise ValueError(f"{study} has no complete screening/confirmation pairs")
    result: list[MarginSummary] = []
    for margin in margins:
        threshold = 1.0 + float(margin)
        screening = sum(screen > threshold for screen, _ in pairs)
        confirmation = sum(confirm > threshold for _, confirm in pairs)
        screen_only = sum(
            screen > threshold and confirm <= threshold for screen, confirm in pairs
        )
        result.append(
            MarginSummary(
                study=study,
                margin=float(margin),
                eligible_winners=len(pairs),
                screening_wins=screening,
                confirmation_wins=confirmation,
                screen_only_promotions=screen_only,
                notes=notes,
            )
        )
    return result


def main() -> int:
    holdout_path = ROOT / (
        "artifacts/workshop2026/holdout_campaign/analysis/holdout_confirmation.csv"
    )
    near_path = ROOT / "reports/tables/workshop2026_near_threshold_winners.csv"
    output_path = ROOT / "reports/tables/workshop2026_margin_sensitivity.csv"
    report_path = ROOT / "reports/workshop2026_margin_sensitivity.md"

    summaries: list[MarginSummary] = []
    summaries.extend(
        summarize_rows(
            "RQ1 KernelBench",
            _read_rows(holdout_path),
            notes="10 frozen valid-task winners",
        )
    )
    summaries.extend(
        summarize_rows(
            "RQ2 near-threshold K=8",
            _read_rows(near_path),
            notes="4 frozen full-budget winners; post-hoc sensitivity",
        )
    )
    _write_csv(output_path, summaries)
    _write_report(report_path, summaries)
    print(f"margin sensitivity table: {output_path}")
    print(f"margin sensitivity report: {report_path}")
    return 0


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[MarginSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MarginSummary.__annotations__))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _write_report(path: Path, rows: list[MarginSummary]) -> None:
    lines = [
        "# Workshop 2026 Promotion-Margin Sensitivity",
        "",
        "This post-hoc analysis changes only the practical promotion threshold; it does not rerun timing or alter the prespecified 2% primary result.",
        "",
        "| Study | Margin | Eligible | Screen | Confirm | Screen-only |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.study} | {row.margin:.0%} | {row.eligible_winners} | "
            f"{row.screening_wins} | {row.confirmation_wins} | "
            f"{row.screen_only_promotions} |"
        )
    lines.extend(
        [
            "",
            "RQ1 remains a null result across 0-5% margins. The near-threshold screen-only promotion appears at 1%, 2%, and 3%; it is absent at parity and when no candidate clears a 5% margin.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
