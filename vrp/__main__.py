"""python -m vrp 统一入口。"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vrp", description="VRP 模拟退火求解器")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("cvrp", help="CVRP 批量求解")
    sub.add_parser("vrptw", help="VRPTW 批量求解")

    args, rest = parser.parse_known_args(argv)
    sys.argv = [f"vrp-{args.command}", *rest]

    if args.command == "cvrp":
        from vrp.cli.run_cvrp import main as run_cvrp

        run_cvrp()
    elif args.command == "vrptw":
        from vrp.cli.run_vrptw import main as run_vrptw

        run_vrptw()


if __name__ == "__main__":
    main()
