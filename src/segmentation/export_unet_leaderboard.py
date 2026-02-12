from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

DB_PATH = Path("outputs/experiments.sqlite")
OUT_PATH = Path("outputs/unet_leaderboard.csv")


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError("DB not found. Run inference first.")

    conn = sqlite3.connect(DB_PATH)

    rows = conn.execute(
        """
        SELECT
          r.created_at,
          r.run_id,
          r.notes,
          MAX(CASE WHEN m.name='dice_vs_fiji' THEN m.value END) AS dice,
          MAX(CASE WHEN m.name='iou_vs_fiji' THEN m.value END)  AS iou,
          MAX(CASE WHEN m.name='pred_defect_ratio' THEN m.value END) AS pred_ratio,
          MAX(CASE WHEN m.name='gt_defect_ratio' THEN m.value END)   AS gt_ratio
        FROM runs r
        JOIN metrics m ON r.run_id = m.run_id
        WHERE r.task = 'segmentation-infer'
          AND r.model_name = 'unet-small'
        GROUP BY r.created_at, r.run_id, r.notes
        ORDER BY dice DESC
        """
    ).fetchall()

    conn.close()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["created_at", "run_id", "sample", "dice_vs_fiji", "iou_vs_fiji", "pred_defect_ratio", "gt_defect_ratio"])
        for created_at, run_id, notes, dice, iou, pr, gr in rows:
            # sample is in notes like: sample=xxx.jpg ...
            sample = ""
            if notes and "sample=" in notes:
                sample = notes.split("sample=")[1].split()[0]
            w.writerow([created_at, run_id, sample, dice, iou, pr, gr])

    print(f"Saved: {OUT_PATH.resolve()}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
