from __future__ import annotations

import csv
from pathlib import Path

from scripts.analyze_lifecycle_uncertainty import summarize_lifecycle_rows


def test_lifecycle_uncertainty_preserves_task_clusters(tmp_path: Path) -> None:
    path = tmp_path / "lifecycle.csv"
    rows = []
    for task_id, host, event in (
        ("a", (1.0, 1.1, 1.2), (0.99, 1.0, 1.01)),
        ("b", (1.2, 1.3, 1.4), (1.0, 1.01, 1.02)),
    ):
        for process_id, (host_value, event_value) in enumerate(zip(host, event, strict=True)):
            rows.append(
                {
                    "task_id": task_id,
                    "process_id": f"p{process_id}",
                    "median_host_lifecycle_inflation": host_value,
                    "median_enclosing_event_inflation": event_value,
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summaries = summarize_lifecycle_rows(path, bootstrap_samples=200, seed=7)

    assert [row["metric"] for row in summaries] == [
        "host_lifecycle_inflation",
        "enclosing_event_inflation",
    ]
    assert summaries[0]["tasks"] == 2
    assert summaries[0]["process_rows"] == 6
    assert summaries[0]["median"] == 1.2
    assert summaries[0]["task_cluster_bootstrap_lo"] <= summaries[0]["median"]
    assert summaries[0]["task_cluster_bootstrap_hi"] >= summaries[0]["median"]
