from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(path: str | Path = "configs/base.yaml") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))
