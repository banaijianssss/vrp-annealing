"""路线度量与公共工具（CVRP / VRPTW 共用）。"""

from __future__ import annotations

import numpy as np

Solution = list[list[int]]


def euclidean_dist_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def route_load(demands: np.ndarray, route: list[int]) -> float:
    return float(demands[route].sum()) if route else 0.0


def route_distance(dist: np.ndarray, depot: int, route: list[int]) -> float:
    if not route:
        return 0.0
    d = float(dist[depot, route[0]])
    for a, b in zip(route, route[1:]):
        d += dist[a, b]
    d += float(dist[route[-1], depot])
    return d


def solution_distance(dist: np.ndarray, depot: int, routes: Solution) -> float:
    return sum(route_distance(dist, depot, r) for r in routes)


def prune_empty(routes: Solution) -> Solution:
    return [r for r in routes if r]
