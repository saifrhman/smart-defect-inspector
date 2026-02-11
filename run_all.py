from __future__ import annotations

import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    # Keep everything using module execution
    run([sys.executable, "-m", "src.utils.smoke_test"])
    run([sys.executable, "-m", "src.data.make_dataset"])
    run([sys.executable, "-m", "src.preprocess.demo_preprocess"])
    run([sys.executable, "-m", "src.db.query_latest_runs"])
    run([sys.executable, "-m", "src.eval.visualize_preprocess"])
    run([sys.executable, "-m", "src.segmentation.make_synthetic_mask"])
    run([sys.executable, "-m", "src.segmentation.visualize_mask_overlay"])

    print("\nFull pipeline complete. Check outputs/ and the SQLite DB.")


if __name__ == "__main__":
    main()
