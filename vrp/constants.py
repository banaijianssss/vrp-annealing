"""全局常量与算例发现。"""

from __future__ import annotations

import re
from pathlib import Path

from vrp.paths import INSTANCE_DIR

_N_PATTERN = re.compile(r"_N(\d+)", re.IGNORECASE)


def _customer_count_from_name(filename: str) -> int:
    match = _N_PATTERN.search(filename)
    return int(match.group(1)) if match else 0


def discover_instance_files(instance_dir: Path | None = None) -> list[str]:
    """扫描算例目录下的 .TXT / .txt 文件，按客户数从小到大排序。"""
    base = instance_dir or INSTANCE_DIR
    if not base.is_dir():
        return []
    files = [p.name for p in base.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
    return sorted(files, key=lambda f: (_customer_count_from_name(f), f))


INSTANCE_FILES = discover_instance_files()