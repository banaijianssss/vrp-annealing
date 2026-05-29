"""VRPTW 批量求解 CLI。"""

from __future__ import annotations

from pathlib import Path

from vrp.cli.batch_run import ProblemSpec, build_arg_parser, run_batch
from vrp.data.instance import VRPTWInstance, load_vrptw_instance
from vrp.paths import VRPTW_DIR
from vrp.sa.vrptw import route_distance, route_load, route_schedule, solve_vrptw_sa
from vrp.viz.routes import plot_vrptw_routes


def format_routes(inst: VRPTWInstance, routes: list[list[int]], dist) -> str:
    lines = []
    for k, route in enumerate(routes, 1):
        load = route_load(inst, route)
        rd = route_distance(dist, inst.depot, route)
        sched = route_schedule(inst, dist, route)
        seq = " -> ".join(["0"] + [str(c) for c in route] + ["0"])
        lines.append(f"  车辆{k}: 载重={load:.0f}/{inst.capacity:.0f}, 距离={rd:.2f}")
        lines.append(f"         路径: {seq}")
        if sched:
            tw_parts = [
                f"{v.customer}[到达{v.arrival:.0f},开始{v.start:.0f},离开{v.depart:.0f}]"
                for v in sched
            ]
            lines.append(f"         时刻: {' | '.join(tw_parts)}")
        else:
            lines.append("         时刻: (路线时间窗不可行)")
    return "\n".join(lines)


def _print_vrptw_extra(inst: VRPTWInstance, result) -> None:
    if result.feasible:
        print(f"  最晚回库时刻: {result.max_route_end_time:.1f} (仓库 due={inst.due[inst.depot]:.0f})")


def _plot_vrptw(name, coords, routes, objective, out_path, *, seed, feasible=None, **_) -> Path:
    return plot_vrptw_routes(name, coords, routes, objective, out_path, seed=seed, feasible=feasible)


SPEC = ProblemSpec(
    problem_id="vrptw",
    title="VRPTW 模拟退火求解（容量 + 时间窗，行驶时间=欧氏距离）",
    subtitle="VRPTW 模拟退火求解（容量 + 时间窗）",
    results_dir=VRPTW_DIR,
    preset_help="求解档位: fast / medium / high / ultra",
    plot_suffix="VRPTW_SA",
    load_instance=load_vrptw_instance,
    solve=solve_vrptw_sa,
    format_routes=format_routes,
    plot_routes=_plot_vrptw,
    solution_extra_lines=lambda result, _seed: [f"最晚回库: {result.max_route_end_time:.2f}"],
    print_extra=_print_vrptw_extra,
    summary_extra=lambda result: {"max_route_end_time": round(result.max_route_end_time, 2)},
)


def main() -> None:
    run_batch(SPEC, build_arg_parser(SPEC).parse_args())


if __name__ == "__main__":
    main()
