from __future__ import annotations

from pathlib import Path
import sqlite3
from datetime import datetime, timezone
import uuid
from typing import Dict, Optional


DB_PATH = Path("outputs/experiments.sqlite")
SCHEMA_PATH = Path("src/db/schema.sql")


def init_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        conn.commit()


def create_run(
    task: str,
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    notes: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> str:
    init_db(db_path)
    run_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO runs (run_id, created_at, task, model_name, dataset_name, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, created_at, task, model_name, dataset_name, notes),
        )
        conn.commit()

    return run_id


def log_metrics(run_id: str, metrics: Dict[str, float], db_path: Path = DB_PATH) -> None:
    init_db(db_path)
    rows = [(run_id, k, float(v)) for k, v in metrics.items()]

    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO metrics (run_id, name, value) VALUES (?, ?, ?)",
            rows,
        )
        conn.commit()


def main() -> None:
    run_id = create_run(task="preprocess", model_name="opencv", dataset_name="synthetic", notes="smoke metrics")
    log_metrics(run_id, {"dummy_score": 1.0})
    print("✅ Logged run:", run_id)
    print(f"DB: {DB_PATH.resolve()}")


if __name__ == "__main__":
    main()
