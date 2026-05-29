"""CVRP 批量求解 CLI。"""

from __future__ import annotations

from pathlib import Path

from vrp.cli.batch_run import ProblemSpec, build_arg_parser, run_batch
from vrp.data.instance import CVRPInstance, load_instance
from vrp.paths import CVRP_DIR
from vrp.sa.cvrp import route_distance, route_load, solve_cvrp_sa
from vrp.viz.routes import plot_cvrp_routes


def format_routes(inst: CVRPInstance, routes: list[list[int]], dist) -> str:
    lines = []
    for k, route in enumerate(routes, 1):
        load = route_load(inst, route)
        rd = route_distance(dist, inst.depot, route)
        seq = " -> ".join(["0"] + [str(c) for c in route] + ["0"])
        lines.append(f"  车辆{k}: 载重={load:.0f}/{inst.capacity:.0f}, 距离={rd:.2f}")
        lines.append(f"         路径: {seq}")
    return "\n".join(lines)


def _plot_cvrp(name, coords, routes, objective, out_path, *, seed, **_) -> Path:
    return plot_cvrp_routes(name, coords, routes, objective, out_path, seed=seed)


SPEC = ProblemSpec(
    problem_id="cvrp",
    title="CVRP 模拟退火求解（无时间窗，仅容量约束）",
    subtitle="CVRP 模拟退火求解（无时间窗，仅容量约束）",
    results_dir=CVRP_DIR,
    preset_help="求解档位: fast / medium / high / ultra(逼近最优，较慢)",
    plot_suffix="CVRP_SA",
    load_instance=load_instance,
    solve=solve_cvrp_sa,
    format_routes=format_routes,
    plot_routes=_plot_cvrp,
    solution_extra_lines=lambda _result, _seed: [],
)


def main() -> None:
    run_batch(SPEC, build_arg_parser(SPEC).parse_args())


if __name__ == "__main__":
    main()
