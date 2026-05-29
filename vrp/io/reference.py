"""参考最优距离（用于 gap 计算）。"""

from __future__ import annotations

import json
from pathlib import Path

from vrp.paths import REFERENCE_DIR

_OPTIMAL_FILE = REFERENCE_DIR / "optimal_distances.json"


def load_reference_distances() -> dict[str, float]:
    if not _OPTIMAL_FILE.is_file():
        return {}
    data = json.loads(_OPTIMAL_FILE.read_text(encoding="utf-8"))
    return {name: float(dist) for name, dist in data.items() if dist is not None}


def gap_pct(distance: float, reference: float | None) -> float | None:
    if reference is None or reference <= 0:
        return None
    return round((distance - reference) / reference * 100, 4)
