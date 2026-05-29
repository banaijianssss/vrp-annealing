"""模拟退火档位参数。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PresetName = Literal["fast", "medium", "high", "ultra"]
PRESET_CHOICES: tuple[str, ...] = ("fast", "medium", "high", "ultra")


@dataclass(frozen=True)
class SAPreset:
    """模拟退火档位参数（随客户数 n 线性缩放）。"""

    label: str
    max_iter_base: int
    max_iter_per_n: int
    iter_per_temp_base: int
    iter_per_temp_per_n: int
    alpha: float
    t_min: float
    t0_factor: float
    no_improve_patience: int
    restarts: int

    def max_iter_for(self, n: int) -> int:
        return self.max_iter_base + self.max_iter_per_n * n

    def iter_per_temp_for(self, n: int) -> int:
        return self.iter_per_temp_base + self.iter_per_temp_per_n * n


PRESETS: dict[PresetName, SAPreset] = {
    "fast": SAPreset(
        label="快速",
        max_iter_base=5_000,
        max_iter_per_n=200,
        iter_per_temp_base=50,
        iter_per_temp_per_n=3,
        alpha=0.99,
        t_min=1e-3,
        t0_factor=0.03,
        no_improve_patience=1_000,
        restarts=1,
    ),
    "medium": SAPreset(
        label="均衡",
        max_iter_base=8_000,
        max_iter_per_n=400,
        iter_per_temp_base=80,
        iter_per_temp_per_n=4,
        alpha=0.995,
        t_min=1e-3,
        t0_factor=0.05,
        no_improve_patience=2_000,
        restarts=1,
    ),
    "high": SAPreset(
        label="高精度",
        max_iter_base=20_000,
        max_iter_per_n=1_000,
        iter_per_temp_base=100,
        iter_per_temp_per_n=10,
        alpha=0.997,
        t_min=1e-3,
        t0_factor=0.05,
        no_improve_patience=3_000,
        restarts=8,
    ),
    "ultra": SAPreset(
        label="逼近最优",
        max_iter_base=40_000,
        max_iter_per_n=2_000,
        iter_per_temp_base=150,
        iter_per_temp_per_n=15,
        alpha=0.998,
        t_min=1e-4,
        t0_factor=0.04,
        no_improve_patience=5_000,
        restarts=15,
    ),
}


def normalize_preset(preset: str) -> PresetName:
    key = preset.strip().lower()
    if key not in PRESETS:
        raise ValueError(f"未知档位 '{preset}'，可选: {', '.join(PRESET_CHOICES)}")
    return key  # type: ignore[return-value]


