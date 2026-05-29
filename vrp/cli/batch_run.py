"""批量求解 CLI 通用框架。"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from vrp.constants import INSTANCE_FILES
from vrp.io.reference import gap_pct, load_reference_distances
from vrp.io.results import write_solution_txt, write_summary_json
from vrp.paths import INSTANCE_DIR, REFERENCE_DIR, ROOT
from vrp.sa.presets import PRESETS, PRESET_CHOICES
from vrp.util.run_output import make_run_output_dir, write_latest_run
from vrp.util.seed import add_seed_argument, resolve_seed

PROJECT_DIR = ROOT


@dataclass(frozen=True)
class ProblemSpec:
    problem_id: str
    title: str
    subtitle: str
    results_dir: Path
    preset_help: str
    plot_suffix: str
    load_instance: Callable[[Path], Any]
    solve: Callable[..., Any]
    format_routes: Callable[[Any, list[list[int]], Any], str]
    plot_routes: Callable[..., Path]
    solution_extra_lines: Callable[[Any, int], list[str]]
    print_extra: Callable[[Any, Any], None] | None = None
    summary_extra: Callable[[Any], dict[str, Any]] | None = None


def build_arg_parser(spec: ProblemSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=spec.subtitle)
    parser.add_argument(
        "--preset",
        "-p",
        choices=PRESET_CHOICES,
        default="medium",
        help=spec.preset_help,
    )
    add_seed_argument(parser)
    return parser


def run_batch(spec: ProblemSpec, args: argparse.Namespace) -> None:
    preset_key = args.preset
    preset_cfg = PRESETS[preset_key]
    seed = resolve_seed(args.seed)
    references = load_reference_distances()

    run_dir = make_run_output_dir(spec.results_dir, seed=seed, preset=preset_key)
    plots_dir = run_dir / "plots"
    write_latest_run(
        spec.results_dir / "latest_run.json",
        problem=spec.problem_id,
        preset=preset_key,
        seed=seed,
        run_dir=run_dir,
    )

    summary: list[dict[str, Any]] = []

    print("=" * 72)
    print(spec.title)
    print(f"算例目录: {INSTANCE_DIR}")
    print(f"输出目录: {run_dir.relative_to(PROJECT_DIR)}")
    print(f"随机种子: {seed}" + ("（指定）" if args.seed is not None else "（自动生成）"))
    print(
        f"档位: {preset_key} ({preset_cfg.label}) | "
        f"max_iter≈{preset_cfg.max_iter_base}+{preset_cfg.max_iter_per_n}·n | "
        f"重启={preset_cfg.restarts}"
    )
    print("=" * 72)

    for fname in INSTANCE_FILES:
        path = INSTANCE_DIR / fname
        inst = spec.load_instance(path)
        dist = inst.dist_matrix()
        n = inst.n_customers
        planned_iter = preset_cfg.max_iter_for(n) * preset_cfg.restarts

        print(f"\n【{inst.name}】 客户数={n}, 车辆上限={inst.num_vehicles}, 容量={inst.capacity}")
        print(
            f"  计划: max_iter={preset_cfg.max_iter_for(n)}, "
            f"iter_per_temp={preset_cfg.iter_per_temp_for(n)}, alpha={preset_cfg.alpha}"
        )
        t0 = time.perf_counter()
        result = spec.solve(inst, preset=preset_key, seed=seed)
        elapsed = time.perf_counter() - t0

        print(f"  总距离: {result.total_distance:.2f}")
        print(f"  使用车辆: {result.num_routes} / {inst.num_vehicles}")
        print(f"  可行性: {'是' if result.feasible else '否'}")
        if spec.print_extra is not None:
            spec.print_extra(inst, result)
        ref_dist = references.get(inst.name)
        gap = gap_pct(result.total_distance, ref_dist)
        if ref_dist is not None and gap is not None:
            print(f"  参考最优: {ref_dist:.4f} | 差距: {gap:+.2f}%")
        print(
            f"  迭代步数: {result.iterations} (上限约 {planned_iter}), "
            f"重启: {result.restarts}, 用时: {elapsed:.2f}s"
        )
        routes_text = spec.format_routes(inst, result.routes, dist)
        print(routes_text)

        plot_path = plots_dir / f"{inst.name}_{spec.plot_suffix}.png"
        spec.plot_routes(
            inst.name,
            inst.coords,
            result.routes,
            result.total_distance,
            plot_path,
            seed=seed,
            **({"feasible": result.feasible} if spec.problem_id == "vrptw" else {}),
        )
        print(f"  路径图: {plot_path.relative_to(PROJECT_DIR)}")

        header = [
            f"算例: {inst.name}",
            f"档位: {result.preset}",
            f"随机种子: {seed}",
            f"总距离: {result.total_distance:.4f}",
            f"车辆数: {result.num_routes}",
            f"可行: {result.feasible}",
            f"重启次数: {result.restarts}",
            *spec.solution_extra_lines(result, seed),
        ]
        if ref_dist is not None:
            header.append(f"参考最优: {ref_dist:.4f}")
            if gap is not None:
                header.append(f"差距(%): {gap:+.4f}")
        write_solution_txt(run_dir / f"{inst.name}_solution.txt", header, routes_text)

        row: dict[str, Any] = {
            "instance": inst.name,
            "preset": result.preset,
            "customers": n,
            "vehicles_limit": inst.num_vehicles,
            "vehicles_used": result.num_routes,
            "total_distance": round(result.total_distance, 4),
            "feasible": result.feasible,
            "restarts": result.restarts,
            "time_sec": round(elapsed, 3),
            "iterations": result.iterations,
            "plot": f"plots/{inst.name}_{spec.plot_suffix}.png",
        }
        if ref_dist is not None:
            row["reference_distance"] = round(ref_dist, 4)
            row["gap_pct"] = gap
        if spec.summary_extra is not None:
            row.update(spec.summary_extra(result))
        summary.append(row)

    write_summary_json(
        run_dir / "summary.json",
        {
            "run_dir": run_dir.name,
            "seed": seed,
            "preset": preset_key,
            "restarts": preset_cfg.restarts,
            "results": summary,
        },
    )

    print("\n" + "=" * 72)
    print("汇总")
    print("-" * 72)
    print(f"{'算例':<14} {'客户':>6} {'距离':>12} {'车辆':>8} {'可行':>6} {'用时(s)':>10}")
    for s in summary:
        print(
            f"{s['instance']:<14} {s['customers']:>6} {s['total_distance']:>12.4f} "
            f"{s['vehicles_used']:>3}/{s['vehicles_limit']:<4} "
            f"{'是' if s['feasible'] else '否':>6} {s['time_sec']:>10.3f}"
        )
    print(f"\n结果已保存至: {run_dir.relative_to(PROJECT_DIR)}")
    print(f"路径图已保存至: {plots_dir.relative_to(PROJECT_DIR)}")
    print(f"参考最优路径图（只读）: {REFERENCE_DIR.relative_to(PROJECT_DIR)}")
