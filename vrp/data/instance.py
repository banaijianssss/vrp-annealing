"""解析 Solomon 格式算例（CVRP 忽略时间窗；VRPTW 使用完整字段）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vrp.sa.metrics import euclidean_dist_matrix


@dataclass
class CVRPInstance:
    name: str
    num_vehicles: int
    capacity: float
    coords: np.ndarray  # (n_nodes, 2)，节点 0 为仓库
    demands: np.ndarray  # (n_nodes,)

    @property
    def n_customers(self) -> int:
        return len(self.demands) - 1

    @property
    def depot(self) -> int:
        return 0

    def dist_matrix(self) -> np.ndarray:
        return euclidean_dist_matrix(self.coords)


@dataclass
class VRPTWInstance:
    """带时间窗的 VRPTW 算例（Solomon 格式，行驶时间=欧氏距离）。"""

    name: str
    num_vehicles: int
    capacity: float
    coords: np.ndarray
    demands: np.ndarray
    ready: np.ndarray
    due: np.ndarray
    service: np.ndarray

    @property
    def n_customers(self) -> int:
        return len(self.demands) - 1

    @property
    def depot(self) -> int:
        return 0

    def dist_matrix(self) -> np.ndarray:
        return euclidean_dist_matrix(self.coords)


def _parse_solomon_lines(path: Path) -> tuple[str, int, float, list[tuple]]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]

    name = lines[0]
    num_vehicles = capacity = None
    customers: list[tuple] = []

    i = 0
    while i < len(lines):
        upper = lines[i].upper()
        if upper.startswith("NUMBER"):
            parts = lines[i + 1].split()
            num_vehicles = int(parts[0])
            capacity = float(parts[1])
            i += 2
            continue
        if upper.startswith("CUST"):
            i += 1
            while i < len(lines) and not lines[i].upper().startswith("CUST"):
                parts = lines[i].split()
                if len(parts) >= 4:
                    cid = int(parts[0])
                    x, y = float(parts[1]), float(parts[2])
                    demand = float(parts[3])
                    ready = float(parts[4]) if len(parts) > 4 else 0.0
                    due = float(parts[5]) if len(parts) > 5 else float("inf")
                    service = float(parts[6]) if len(parts) > 6 else 0.0
                    customers.append((cid, x, y, demand, ready, due, service))
                i += 1
            continue
        i += 1

    if num_vehicles is None or not customers:
        raise ValueError(f"无法解析算例: {path}")
    return name, num_vehicles, capacity, customers


def load_instance(path: str | Path) -> CVRPInstance:
    name, num_vehicles, capacity, customers = _parse_solomon_lines(Path(path))
    customers.sort(key=lambda t: t[0])
    n = len(customers)
    coords = np.zeros((n, 2))
    demands = np.zeros(n)
    for cid, x, y, d, *_ in customers:
        coords[cid] = (x, y)
        demands[cid] = d

    return CVRPInstance(
        name=name,
        num_vehicles=num_vehicles,
        capacity=capacity,
        coords=coords,
        demands=demands,
    )


def load_vrptw_instance(path: str | Path) -> VRPTWInstance:
    name, num_vehicles, capacity, customers = _parse_solomon_lines(Path(path))
    customers.sort(key=lambda t: t[0])
    n = len(customers)
    coords = np.zeros((n, 2))
    demands = np.zeros(n)
    ready = np.zeros(n)
    due = np.zeros(n)
    service = np.zeros(n)
    for cid, x, y, d, r, du, s in customers:
        coords[cid] = (x, y)
        demands[cid] = d
        ready[cid] = r
        due[cid] = du
        service[cid] = s

    return VRPTWInstance(
        name=name,
        num_vehicles=num_vehicles,
        capacity=capacity,
        coords=coords,
        demands=demands,
        ready=ready,
        due=due,
        service=service,
    )
