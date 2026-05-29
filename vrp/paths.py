"""项目路径常量。"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INSTANCE_DIR = ROOT / "算例"
REFERENCE_DIR = ROOT / "reference"
RESULTS_DIR = ROOT / "results"
CVRP_DIR = RESULTS_DIR / "cvrp"
VRPTW_DIR = RESULTS_DIR / "vrptw"
