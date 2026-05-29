"""VRPTW 模拟退火求解器。"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from vrp.data.instance import VRPTWInstance
from vrp.sa import vrptw_heuristics as h
from vrp.sa.engine import SAParams, run_sa_once
from vrp.sa.metrics import Solution
from vrp.sa.presets import PRESETS, PRESET_CHOICES, PresetName, normalize_preset

EPS = h.EPS


@dataclass
class SAResult:
    routes: Solution
    total_distance: float
    num_routes: int
    feasible: bool
    iterations: int
    preset: PresetName = "medium"
    restarts: int = 1
    max_route_end_time: float = 0.0


def relocate_neighbor(
    inst: VRPTWInstance, dist: np.ndarray, routes: Solution, rng: random.Random
) -> Solution | None:
    routes = [r[:] for r in routes]
    non_empty = [(i, r) for i, r in enumerate(routes) if r]
    if not non_empty:
        return None
    ri, route_from = rng.choice(non_empty)
    pos_from = rng.randrange(len(route_from))
    customer = route_from.pop(pos_from)
    if not route_from:
        routes.pop(ri)
    else:
        routes[ri] = route_from

    candidates = list(enumerate(routes))
    if len(h.prune_empty(routes)) < inst.num_vehicles:
        candidates.append((len(routes), []))

    rng.shuffle(candidates)
    for rj, route_to in candidates:
        if h.route_load(inst, route_to) + inst.demands[customer] > inst.capacity:
            continue
        trial = route_to[:]
        positions = list(range(len(trial) + 1))
        rng.shuffle(positions)
        for pos in positions:
            trial.insert(pos, customer)
            if rj < len(routes):
                routes[rj] = trial
            else:
                routes.append(trial)
            if h.is_feasible(inst, dist, routes):
                return h.prune_empty(routes)
            if rj < len(routes):
                routes[rj] = route_to[:]
            else:
                routes.pop()
            trial = route_to[:]

    routes = [r[:] for r in routes]
    if ri >= len(routes):
        routes.append([])
    routes[ri].insert(pos_from, customer)
    return h.prune_empty(routes) if h.is_feasible(inst, dist, routes) else None


def swap_neighbor(inst: VRPTWInstance, dist: np.ndarray, routes: Solution, rng: random.Random) -> Solution | None:
    routes = [r[:] for r in routes]
    non_empty = [i for i, r in enumerate(routes) if r]
    if not non_empty:
        return None
    if len(non_empty) == 1:
        r = routes[non_empty[0]]
        if len(r) < 2:
            return None
        i, j = rng.sample(range(len(r)), 2)
        r[i], r[j] = r[j], r[i]
        return h.prune_empty(routes) if h.is_feasible(inst, dist, routes) else None

    i, j = rng.sample(non_empty, 2)
    ri, rj = routes[i], routes[j]
    pi, pj = rng.randrange(len(ri)), rng.randrange(len(rj))
    ri[pi], rj[pj] = rj[pj], ri[pi]
    if not h.is_feasible(inst, dist, routes):
        return None
    return h.prune_empty(routes)


def two_opt_neighbor(
    inst: VRPTWInstance, dist: np.ndarray, routes: Solution, rng: random.Random
) -> Solution | None:
    routes = [r[:] for r in routes]
    long_routes = [i for i, r in enumerate(routes) if len(r) >= 4]
    if not long_routes:
        return None
    ri = rng.choice(long_routes)
    r = routes[ri]
    i = rng.randint(0, len(r) - 3)
    j = rng.randint(i + 2, len(r) - 1)
    new_route = r[: i + 1] + r[i + 1 : j + 1][::-1] + r[j + 1 :]
    if not h.is_route_feasible(inst, dist, new_route):
        return None
    routes[ri] = new_route
    return h.prune_empty(routes)


def random_neighbor(
    inst: VRPTWInstance, dist: np.ndarray, routes: Solution, rng: random.Random
) -> Solution | None:
    ops = [
        lambda: relocate_neighbor(inst, dist, routes, rng),
        lambda: swap_neighbor(inst, dist, routes, rng),
        lambda: two_opt_neighbor(inst, dist, routes, rng),
    ]
    rng.shuffle(ops)
    for op in ops:
        cand = op()
        if cand is not None:
            return cand
    return None


def _solve_vrptw_sa_once(
    inst: VRPTWInstance,
    dist: np.ndarray,
    *,
    seed: int,
    max_iter: int,
    iter_per_temp: int,
    t0: float | None,
    t_min: float,
    t0_factor: float,
    alpha: float,
    no_improve_patience: int,
) -> SAResult:
    n = inst.n_customers
    params = SAParams(
        max_iter=max_iter,
        iter_per_temp=iter_per_temp,
        t0=t0,
        t_min=t_min,
        t0_factor=t0_factor,
        alpha=alpha,
        no_improve_patience=no_improve_patience,
        stall_cool_factor=0.9,
        cost_eps=EPS,
    )
    step = run_sa_once(
        seed=seed,
        params=params,
        build_initial=lambda rng: h.local_search(
            inst, dist, h.build_initial_solution(inst, dist, rng, seed)
        ),
        fallback_initial=lambda rng: (
            h.local_search(inst, dist, fb)
            if (fb := h.prune_empty(h.cvrp_warm_start(inst, dist, seed)))
            else [[i] for i in range(1, n + 1)]
        ),
        random_neighbor=lambda routes, rng: random_neighbor(inst, dist, routes, rng),
        local_search=lambda routes: h.local_search(inst, dist, routes),
        finalize=lambda routes: h.local_search(inst, dist, routes),
        cost=lambda routes: h.solution_distance(inst, dist, routes),
        is_feasible=lambda routes: h.is_feasible(inst, dist, routes),
    )
    feasible = step.feasible
    return SAResult(
        routes=step.routes,
        total_distance=step.cost,
        num_routes=len(step.routes),
        feasible=feasible,
        iterations=step.iterations,
        max_route_end_time=h.max_solution_end_time(inst, dist, step.routes) if feasible else math.inf,
    )


def solve_vrptw_sa(
    inst: VRPTWInstance,
    *,
    preset: PresetName | str = "medium",
    seed: int = 42,
    max_iter: int | None = None,
    t0: float | None = None,
    t_min: float | None = None,
    alpha: float | None = None,
    iter_per_temp: int | None = None,
    restarts: int | None = None,
) -> SAResult:
    preset_key = normalize_preset(preset)
    cfg = PRESETS[preset_key]
    n = inst.n_customers
    dist = inst.dist_matrix()

    run_max_iter = max_iter if max_iter is not None else cfg.max_iter_for(n)
    run_iter_per_temp = iter_per_temp if iter_per_temp is not None else cfg.iter_per_temp_for(n)
    run_t_min = t_min if t_min is not None else cfg.t_min
    run_alpha = alpha if alpha is not None else cfg.alpha
    run_restarts = restarts if restarts is not None else cfg.restarts

    best_result: SAResult | None = None
    total_iterations = 0

    for run in range(run_restarts):
        run_seed = seed + run
        result = _solve_vrptw_sa_once(
            inst,
            dist,
            seed=run_seed,
            max_iter=run_max_iter,
            iter_per_temp=run_iter_per_temp,
            t0=t0,
            t_min=run_t_min,
            t0_factor=cfg.t0_factor,
            alpha=run_alpha,
            no_improve_patience=cfg.no_improve_patience,
        )
        total_iterations += result.iterations
        if best_result is None or (
            result.feasible
            and (
                not best_result.feasible
                or result.total_distance < best_result.total_distance - EPS
            )
        ):
            best_result = result
        elif not best_result.feasible and result.total_distance < best_result.total_distance - EPS:
            best_result = result

    assert best_result is not None
    return SAResult(
        routes=best_result.routes,
        total_distance=best_result.total_distance,
        num_routes=best_result.num_routes,
        feasible=best_result.feasible,
        iterations=total_iterations,
        preset=preset_key,
        restarts=run_restarts,
        max_route_end_time=best_result.max_route_end_time,
    )


VisitInfo = h.VisitInfo
route_load = h.route_load
route_distance = h.route_distance
route_schedule = h.route_schedule
is_feasible = h.is_feasible
prune_empty = h.prune_empty

__all__ = [
    "PRESET_CHOICES",
    "SAResult",
    "VisitInfo",
    "is_feasible",
    "route_distance",
    "route_load",
    "route_schedule",
    "solve_vrptw_sa",
]
