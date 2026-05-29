"""通用模拟退火引擎。"""

from __future__ import annotations

import math
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import numpy as np

from vrp.sa.metrics import Solution

BuildInitial = Callable[[random.Random], Solution]
FallbackInitial = Callable[[random.Random], Solution]
NeighborFn = Callable[[Solution, random.Random], Solution | None]
LocalSearchFn = Callable[[Solution], Solution]
FinalizeFn = Callable[[Solution], Solution]
CostFn = Callable[[Solution], float]
FeasibleFn = Callable[[Solution], bool]


@dataclass(frozen=True)
class SAParams:
    max_iter: int
    iter_per_temp: int
    t0: float | None
    t_min: float
    t0_factor: float
    alpha: float
    no_improve_patience: int
    stall_cool_factor: float = 0.85
    cost_eps: float = 1e-9


@dataclass
class SAStepResult:
    routes: Solution
    cost: float
    iterations: int
    feasible: bool


def run_sa_once(
    *,
    seed: int,
    params: SAParams,
    build_initial: BuildInitial,
    fallback_initial: FallbackInitial,
    random_neighbor: NeighborFn,
    local_search: LocalSearchFn,
    finalize: FinalizeFn,
    cost: CostFn,
    is_feasible: FeasibleFn,
) -> SAStepResult:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    current = build_initial(rng)
    if not is_feasible(current):
        current = fallback_initial(rng)

    current_cost = cost(current)
    best = deepcopy(current)
    best_cost = current_cost

    t0 = params.t0
    if t0 is None:
        t0 = max(1.0, params.t0_factor * current_cost)

    temperature = t0
    total_steps = 0
    no_improve = 0

    while temperature > params.t_min and total_steps < params.max_iter:
        for _ in range(params.iter_per_temp):
            total_steps += 1
            neighbor = random_neighbor(current, rng)
            if neighbor is None:
                continue
            neighbor = local_search(neighbor)
            if not is_feasible(neighbor):
                continue

            neighbor_cost = cost(neighbor)
            delta = neighbor_cost - current_cost

            if delta < 0 or np_rng.random() < math.exp(-delta / temperature):
                current, current_cost = neighbor, neighbor_cost
                if neighbor_cost < best_cost - params.cost_eps:
                    best, best_cost = deepcopy(neighbor), neighbor_cost
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

            if total_steps >= params.max_iter:
                break

        temperature *= params.alpha
        if no_improve > params.no_improve_patience:
            temperature *= params.stall_cool_factor
            no_improve = 0

    best = finalize(best)
    best_cost = cost(best)
    return SAStepResult(
        routes=best,
        cost=best_cost,
        iterations=total_steps,
        feasible=is_feasible(best),
    )
