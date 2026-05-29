"""随机种子 CLI 参数与解析。"""

from __future__ import annotations

import argparse
import secrets


def add_seed_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子；不指定则每次运行自动生成（可用 --seed 123 复现）",
    )


def resolve_seed(cli_seed: int | None) -> int:
    if cli_seed is not None:
        return cli_seed
    return secrets.randbelow(2_147_483_647)
