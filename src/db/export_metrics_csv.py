from __future__ import annotations

import sqlite3
from pathlib import Path
import csv

DB_PATH = Path("outputs/experiments.sqlite")
OUT_PATH = Path("outputs/metrics_export.csv")


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError("DB not found. Run the pipeline first.")

    conn = sqlite3.connect(DB_PATH)

    rows = conn.execute(
        """
        SELECT
          r.run_id,
          r.created_at,
          r.task,
          r.model_name,
          r.dataset_name,
          COALESCE(r.notes, '') AS notes,
          m.name AS metric_name,
          m.value AS metric_value
        FROM runs r
        JOIN metrics m ON r.run_id = m.run_id
        ORDER BY r.created_at DESC
        """
    ).fetchall()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "created_at", "task", "model_name", "dataset_name", "notes", "metric_name", "metric_value"])
        w.writerows(rows)

    conn.close()
    print(f"Saved: {OUT_PATH.resolve()}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
