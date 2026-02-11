from __future__ import annotations

from pathlib import Path
import sqlite3
import matplotlib.pyplot as plt


DB_PATH = Path("outputs/experiments.sqlite")


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError("DB not found. Run pipeline first.")

    conn = sqlite3.connect(DB_PATH)

    # Pull defect_ratio metrics from Fiji batch runs
    rows = conn.execute(
        """
        SELECT r.created_at, r.notes, m.value
        FROM runs r
        JOIN metrics m ON r.run_id = m.run_id
        WHERE r.task LIKE 'fiji-segmentation%'
          AND m.name = 'defect_ratio'
        ORDER BY r.created_at ASC
        """
    ).fetchall()

    conn.close()

    if not rows:
        raise RuntimeError("No fiji defect_ratio metrics found. Run: python -m src.fiji.batch_log_fiji_masks")

    values = [float(r[2]) for r in rows]

    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=20)
    plt.title("Fiji mask defect_ratio distribution")
    plt.xlabel("defect_ratio")
    plt.ylabel("count")

    out_path = Path("outputs/defect_ratio_hist.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved: {out_path.resolve()}")
    print(f"n={len(values)} | min={min(values):.4f} | max={max(values):.4f} | mean={sum(values)/len(values):.4f}")


if __name__ == "__main__":
    main()
