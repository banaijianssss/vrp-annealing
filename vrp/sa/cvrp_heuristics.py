"""CVRP 构造启发式与强化局部搜索（供模拟退火调用）。"""

from __future__ import annotations

import math
import random

import numpy as np

from vrp.data.instance import CVRPInstance
from vrp.sa.metrics import (
    Solution,
    prune_empty,
    route_distance,
    route_load as _route_load,
    solution_distance as _solution_distance,
)

EPS = 1e-9


def route_load(inst: CVRPInstance, route: list[int]) -> float:
    return _route_load(inst.demands, route)


def solution_distance(inst: CVRPInstance, dist: np.ndarray, routes: Solution) -> float:
    return _solution_distance(dist, inst.depot, routes)


def is_feasible(inst: CVRPInstance, routes: Solution) -> bool:
    visited = [c for r in routes for c in r]
    if len(visited) != len(set(visited)):
        return False
    if set(visited) != set(range(1, inst.n_customers + 1)):
        return False
    if len([r for r in routes if r]) > inst.num_vehicles:
        return False
    return all(route_load(inst, r) <= inst.capacity + EPS for r in routes if r)


def two_opt_route(dist: np.ndarray, depot: int, route: list[int]) -> list[int]:
    if len(route) < 4:
        return route[:]
    best = route[:]
    best_cost = route_distance(dist, depot, best)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 2):
            for j in range(i + 2, len(best)):
                new_route = best[: i + 1] + best[i + 1 : j + 1][::-1] + best[j + 1 :]
                new_cost = route_distance(dist, depot, new_route)
                if new_cost + EPS < best_cost:
                    best, best_cost = new_route, new_cost
                    improved = True
    return best


def or_opt_route(dist: np.ndarray, depot: int, route: list[int], max_chain: int = 3) -> list[int]:
    """路线内 Or-opt：移动长度 1~max_chain 的连续客户段。"""
    if len(route) < 2:
        return route[:]
    best = route[:]
    best_cost = route_distance(dist, depot, best)
    improved = True
    while improved:
        improved = False
        n = len(best)
        for length in range(1, min(max_chain, n) + 1):
            for start in range(n - length + 1):
                chain = best[start : start + length]
                rest = best[:start] + best[start + length :]
                for pos in range(len(rest) + 1):
                    trial = rest[:pos] + chain + rest[pos:]
                    cost = route_distance(dist, depot, trial)
                    if cost + EPS < best_cost:
                        best, best_cost = trial, cost
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return best


def cheapest_insertion_init(inst: CVRPInstance, dist: np.ndarray, rng: random.Random) -> Solution:
    unvisited = list(range(1, inst.n_customers + 1))
    rng.shuffle(unvisited)
    routes: Solution = []
    depot = inst.depot

    for customer in unvisited:
        best: tuple[float, int, int] | None = None
        for ri, route in enumerate(routes):
            if route_load(inst, route) + inst.demands[customer] > inst.capacity:
                continue
            for pos in range(len(route) + 1):
                trial = route[:pos] + [customer] + route[pos:]
                cost = route_distance(dist, depot, trial)
                if best is None or cost < best[0]:
                    best = (cost, ri, pos)
        if best is not None:
            routes[best[1]].insert(best[2], customer)
        elif len(prune_empty(routes)) < inst.num_vehicles:
            routes.append([customer])
        else:
            ri = min(range(len(routes)), key=lambda i: route_load(inst, routes[i]))
            routes[ri].append(customer)
    return prune_empty(routes)


def clarke_wright_init(inst: CVRPInstance, dist: np.ndarray, rng: random.Random) -> Solution:
    """Clarke-Wright 节约算法构造初始解。"""
    depot = inst.depot
    customers = list(range(1, inst.n_customers + 1))
    routes: list[list[int]] = [[c] for c in customers]
    route_id = {c: i for i, c in enumerate(customers)}

    savings: list[tuple[float, int, int]] = []
    for i in customers:
        for j in customers:
            if i < j:
                s = dist[depot, i] + dist[depot, j] - dist[i, j]
                savings.append((float(s), i, j))
    savings.sort(reverse=True)
    rng.shuffle(savings)

    def try_merge(ra: list[int], rb: list[int]) -> list[int] | None:
        opts: list[list[int]] = [
            ra + rb,
            ra + rb[::-1],
            ra[::-1] + rb,
            ra[::-1] + rb[::-1],
        ]
        feasible = [o for o in opts if route_load(inst, o) <= inst.capacity + EPS]
        if not feasible:
            return None
        return min(feasible, key=lambda r: route_distance(dist, depot, r))

    for _, i, j in savings:
        if len(routes) <= inst.num_vehicles:
            break
        ri, rj = route_id.get(i), route_id.get(j)
        if ri is None or rj is None or ri == rj:
            continue
        merged = try_merge(routes[ri], routes[rj])
        if merged is None:
            continue
        keep, drop = (ri, rj) if ri < rj else (rj, ri)
        routes[keep] = merged
        routes.pop(drop)
        route_id = {c: idx for idx, r in enumerate(routes) for c in r}

    return prune_empty(routes) if routes else cheapest_insertion_init(inst, dist, rng)


def light_local_search(inst: CVRPInstance, dist: np.ndarray, routes: Solution) -> Solution:
    """路线内 2-opt + Or-opt（SA 内每次邻域后调用，较快）。"""
    depot = inst.depot
    return prune_empty([or_opt_route(dist, depot, two_opt_route(dist, depot, r[:])) for r in routes])


def _relocate_pass(inst: CVRPInstance, dist: np.ndarray, routes: Solution) -> Solution | None:
    """单轮：找一个最优单点 relocate，有改进则返回新解。"""
    base = solution_distance(inst, dist, routes)
    best_cost = base
    best_routes: Solution | None = None
    for ri, route in enumerate(routes):
        for pos, customer in enumerate(route):
            remnant = route[:pos] + route[pos + 1 :]
            trial_base = [r[:] for r in routes]
            trial_base[ri] = remnant
            if not remnant:
                trial_base.pop(ri)
            for rj, rdest in enumerate(trial_base):
                if route_load(inst, rdest) + inst.demands[customer] > inst.capacity:
                    continue
                for ins in range(len(rdest) + 1):
                    tr = [r[:] for r in trial_base]
                    if rj >= len(tr):
                        tr.append([])
                    tr[rj] = tr[rj][:ins] + [customer] + tr[rj][ins:]
                    if not is_feasible(inst, tr):
                        continue
                    cost = solution_distance(inst, dist, tr)
                    if cost + EPS < best_cost:
                        best_cost, best_routes = cost, prune_empty(tr)
            if len(prune_empty(routes)) < inst.num_vehicles:
                tr = [r[:] for r in trial_base]
                tr.append([customer])
                if is_feasible(inst, tr):
                    cost = solution_distance(inst, dist, tr)
                    if cost + EPS < best_cost:
                        best_cost, best_routes = cost, prune_empty(tr)
    if best_routes is not None and best_cost + EPS < base:
        return best_routes
    return None


def intensify_local_search(inst: CVRPInstance, dist: np.ndarray, routes: Solution) -> Solution:
    """强化局部搜索：轻量路线内搜索 + 多轮最优 relocate（用于初解/终解）。"""
    routes = light_local_search(inst, dist, routes)
    for _ in range(max(3, inst.n_customers // 5)):
        moved = _relocate_pass(inst, dist, routes)
        if moved is None:
            break
        routes = light_local_search(inst, dist, moved)
    return routes


def build_initial_solution(inst: CVRPInstance, dist: np.ndarray, rng: random.Random) -> Solution:
    """多种构造启发式取局部搜索后的最优可行解。"""
    builders = [
        lambda: clarke_wright_init(inst, dist, rng),
        lambda: cheapest_insertion_init(inst, dist, rng),
        lambda: cheapest_insertion_init(inst, dist, random.Random(rng.randint(0, 2**31))),
    ]
    best: Solution | None = None
    best_cost = math.inf
    for build in builders:
        routes = prune_empty(build())
        if not is_feasible(inst, routes):
            continue
        routes = intensify_local_search(inst, dist, routes)
        cost = solution_distance(inst, dist, routes)
        if cost < best_cost:
            best, best_cost = routes, cost
    if best is not None:
        return best
    return intensify_local_search(inst, dist, cheapest_insertion_init(inst, dist, rng))


def cross_exchange_neighbor(
    inst: CVRPInstance,
    dist: np.ndarray,
    routes: Solution,
    rng: random.Random,
) -> Solution | None:
    """两条路线间交换尾部片段（类似 2-opt*）。"""
    routes = [r[:] for r in routes]
    non_empty = [i for i, r in enumerate(routes) if len(r) >= 2]
    if len(non_empty) < 2:
        return None
    i, j = rng.sample(non_empty, 2)
    ri, rj = routes[i], routes[j]
    if len(ri) < 2 or len(rj) < 2:
        return None
    cut_i = rng.randint(1, len(ri) - 1)
    cut_j = rng.randint(1, len(rj) - 1)
    new_i = ri[:cut_i] + rj[cut_j:]
    new_j = rj[:cut_j] + ri[cut_i:]
    if route_load(inst, new_i) > inst.capacity or route_load(inst, new_j) > inst.capacity:
        return None
    routes[i], routes[j] = new_i, new_j
    if not is_feasible(inst, routes):
        return None
    return prune_empty(routes)


def or_opt_relocate_neighbor(
    inst: CVRPInstance,
    dist: np.ndarray,
    routes: Solution,
    rng: random.Random,
) -> Solution | None:
    """将一条路线中连续 1~3 个客户移到另一条路线。"""
    routes = [r[:] for r in routes]
    non_empty = [(i, r) for i, r in enumerate(routes) if r]
    if not non_empty:
        return None
    ri, route_from = rng.choice(non_empty)
    if not route_from:
        return None
    max_len = min(3, len(route_from))
    length = rng.randint(1, max_len)
    start = rng.randrange(0, len(route_from) - length + 1)
    chain = route_from[start : start + length]
    routes[ri] = route_from[:start] + route_from[start + length :]
    if not routes[ri]:
        routes.pop(ri)

    candidates = list(enumerate(routes))
    if len(prune_empty(routes)) < inst.num_vehicles:
        candidates.append((len(routes), []))
    rng.shuffle(candidates)
    chain_load = float(inst.demands[chain].sum())
    for rj, route_to in candidates:
        if route_load(inst, route_to) + chain_load > inst.capacity:
            continue
        for pos in range(len(route_to) + 1):
            trial = [r[:] for r in routes]
            if rj < len(trial):
                trial[rj] = route_to[:pos] + chain + route_to[pos:]
            else:
                trial.append(chain[:])
            if is_feasible(inst, trial):
                return prune_empty(trial)
    return None
