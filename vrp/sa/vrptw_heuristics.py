"""VRPTW 构造启发式与局部搜索（供模拟退火调用）。"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from vrp.data.instance import CVRPInstance, VRPTWInstance
from vrp.sa.cvrp import solve_cvrp_sa
from vrp.sa.metrics import (
    Solution,
    prune_empty,
    route_distance,
    route_load as _route_load,
    solution_distance as _solution_distance,
)

EPS = 1e-6


@dataclass
class VisitInfo:
    customer: int
    arrival: float
    start: float
    depart: float


def route_load(inst: VRPTWInstance, route: list[int]) -> float:
    return _route_load(inst.demands, route)


def solution_distance(inst: VRPTWInstance, dist: np.ndarray, routes: Solution) -> float:
    return _solution_distance(dist, inst.depot, routes)


def route_schedule(inst: VRPTWInstance, dist: np.ndarray, route: list[int]) -> list[VisitInfo] | None:
    """按 Solomon 规则模拟一条路线；不可行时返回 None。"""
    if route_load(inst, route) > inst.capacity + EPS:
        return None

    visits: list[VisitInfo] = []
    t = 0.0
    prev = inst.depot
    for c in route:
        t += dist[prev, c]
        arrival = t
        if arrival > inst.due[c] + EPS:
            return None
        start = max(arrival, inst.ready[c])
        depart = start + inst.service[c]
        visits.append(VisitInfo(c, arrival, start, depart))
        t = depart
        prev = c

    t += dist[prev, inst.depot]
    if t > inst.due[inst.depot] + EPS:
        return None
    return visits


def route_end_time(inst: VRPTWInstance, dist: np.ndarray, route: list[int]) -> float:
    sched = route_schedule(inst, dist, route)
    if sched is None:
        return math.inf
    return sched[-1].depart + dist[route[-1], inst.depot]


def is_route_feasible(inst: VRPTWInstance, dist: np.ndarray, route: list[int]) -> bool:
    return route_schedule(inst, dist, route) is not None


def is_feasible(inst: VRPTWInstance, dist: np.ndarray, routes: Solution) -> bool:
    visited = [c for r in routes for c in r]
    if len(visited) != len(set(visited)):
        return False
    expected = set(range(1, inst.n_customers + 1))
    if set(visited) != expected:
        return False
    non_empty = [r for r in routes if r]
    if len(non_empty) > inst.num_vehicles:
        return False
    for r in routes:
        if r and not is_route_feasible(inst, dist, r):
            return False
    return True


def max_solution_end_time(inst: VRPTWInstance, dist: np.ndarray, routes: Solution) -> float:
    return max((route_end_time(inst, dist, r) for r in routes if r), default=0.0)


def cvrp_warm_start(inst: VRPTWInstance, dist: np.ndarray, seed: int) -> Solution:
    """用 CVRP 可行解作为 VRPTW 初解（再经时间窗检查）。"""
    cvrp_inst = CVRPInstance(
        name=inst.name,
        num_vehicles=inst.num_vehicles,
        capacity=inst.capacity,
        coords=inst.coords,
        demands=inst.demands,
    )
    cvrp_result = solve_cvrp_sa(cvrp_inst, preset="fast", seed=seed)
    return prune_empty(cvrp_result.routes)


def time_oriented_init(inst: VRPTWInstance, dist: np.ndarray, rng: random.Random) -> Solution:
    """按 ready 时间排序 + 可行最便宜插入构造初始解。"""
    unvisited = list(range(1, inst.n_customers + 1))
    unvisited.sort(key=lambda c: (inst.ready[c], inst.due[c]))
    rng.shuffle(unvisited)
    routes: Solution = []

    def best_insertion(route: list[int], customer: int) -> tuple[float, int] | None:
        best_cost, best_pos = math.inf, -1
        for pos in range(len(route) + 1):
            trial = route[:pos] + [customer] + route[pos:]
            if not is_route_feasible(inst, dist, trial):
                continue
            cost = route_distance(dist, inst.depot, trial)
            if cost < best_cost:
                best_cost, best_pos = cost, pos
        if best_pos < 0:
            return None
        return best_cost, best_pos

    for customer in unvisited:
        best_global: tuple[float, int, int] | None = None
        for ri, route in enumerate(routes):
            ins = best_insertion(route, customer)
            if ins and (best_global is None or ins[0] < best_global[0]):
                best_global = (ins[0], ri, ins[1])

        if best_global is not None:
            routes[best_global[1]].insert(best_global[2], customer)
        elif len(prune_empty(routes)) < inst.num_vehicles:
            trial = [customer]
            if is_route_feasible(inst, dist, trial):
                routes.append(trial)
            else:
                routes.append([customer])
        else:
            best_ri, best_pos, best_cost = 0, 0, math.inf
            for ri, route in enumerate(routes):
                for pos in range(len(route) + 1):
                    trial = route[:pos] + [customer] + route[pos:]
                    if is_route_feasible(inst, dist, trial):
                        cost = route_distance(dist, inst.depot, trial)
                        if cost < best_cost:
                            best_cost, best_ri, best_pos = cost, ri, pos
            if best_cost < math.inf:
                routes[best_ri].insert(best_pos, customer)
            else:
                ri = min(range(len(routes)), key=lambda i: route_load(inst, routes[i]))
                routes[ri].append(customer)

    return prune_empty(routes) if routes else [[c for c in unvisited]]


def build_initial_solution(inst: VRPTWInstance, dist: np.ndarray, rng: random.Random, seed: int) -> Solution:
    candidates: list[Solution] = []
    warm = prune_empty(cvrp_warm_start(inst, dist, seed))
    if warm:
        candidates.append(warm)
    candidates.append(prune_empty(time_oriented_init(inst, dist, rng)))

    feasible = [r for r in candidates if is_feasible(inst, dist, r)]
    if feasible:
        return min(feasible, key=lambda r: solution_distance(inst, dist, r))

    return min(
        candidates,
        key=lambda r: (len(r), solution_distance(inst, dist, r)),
    )


def two_opt_route(inst: VRPTWInstance, dist: np.ndarray, route: list[int]) -> list[int]:
    if len(route) < 4:
        return route
    best = route[:]
    best_cost = route_distance(dist, inst.depot, best)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 2):
            for j in range(i + 2, len(best)):
                new_route = best[: i + 1] + best[i + 1 : j + 1][::-1] + best[j + 1 :]
                if not is_route_feasible(inst, dist, new_route):
                    continue
                new_cost = route_distance(dist, inst.depot, new_route)
                if new_cost + EPS < best_cost:
                    best, best_cost = new_route, new_cost
                    improved = True
    return best


def local_search(inst: VRPTWInstance, dist: np.ndarray, routes: Solution) -> Solution:
    routes = [two_opt_route(inst, dist, r[:]) for r in routes]
    return prune_empty(routes)
