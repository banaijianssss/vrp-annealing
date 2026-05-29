"""运行输出目录与元数据。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def make_run_output_dir(base_dir: Path, *, seed: int, preset: str) -> Path:
    """创建单次运行目录：{时间}_seed{种子}_{档位}/，内含 plots/ 子目录。"""
    base_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_seed{seed}_{preset}"
    run_dir = base_dir / name
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f"{name}_{suffix}"
        suffix += 1
    run_dir.mkdir()
    (run_dir / "plots").mkdir()
    return run_dir


def write_latest_run(
    path: Path,
    *,
    problem: str,
    preset: str,
    seed: int,
    run_dir: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "problem": problem,
        "preset": preset,
        "seed": seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if run_dir is not None:
        payload["run_dir"] = run_dir.name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
