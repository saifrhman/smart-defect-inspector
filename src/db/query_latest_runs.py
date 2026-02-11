from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path("outputs/experiments.sqlite")


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError("DB not found. Run: python -m src.preprocess.demo_preprocess")

    conn = sqlite3.connect(DB_PATH)

    # Latest runs
    runs = conn.execute(
        """
        SELECT run_id, created_at, task, model_name, dataset_name, COALESCE(notes, '')
        FROM runs
        ORDER BY created_at DESC
        LIMIT 10
        """
    ).fetchall()

    print("\n=== Latest runs (top 10) ===")
    for r in runs:
        run_id, created_at, task, model_name, dataset_name, notes = r
        print(f"- {created_at} | {task} | {model_name} | {dataset_name}")
        print(f"  run_id: {run_id}")
        if notes:
            print(f"  notes: {notes}")

    # Metrics for the most recent run
    if runs:
        latest_run_id = runs[0][0]
        metrics = conn.execute(
            """
            SELECT name, value
            FROM metrics
            WHERE run_id = ?
            ORDER BY name ASC
            """,
            (latest_run_id,),
        ).fetchall()

        print("\n=== Metrics for latest run ===")
        for name, value in metrics:
            print(f"- {name}: {value:.4f}")

    conn.close()


if __name__ == "__main__":
    main()
