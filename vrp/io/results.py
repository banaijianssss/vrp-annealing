"""求解结果文件写入。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_solution_txt(path: Path, header_lines: list[str], routes_text: str) -> None:
    path.write_text("\n".join([*header_lines, "", routes_text]), encoding="utf-8")


def write_summary_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
