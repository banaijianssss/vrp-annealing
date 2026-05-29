"""路径示意图绘制（CVRP / VRPTW，供 main 与可选脚本复用）。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROUTE_COLORS = [
    "#1f4e79",
    "#5b9bd5",
    "#ed7d31",
    "#c55a11",
    "#548235",
    "#7030a0",
    "#00b0f0",
    "#ffc000",
    "#7f7f7f",
    "#264478",
    "#9e480e",
    "#636363",
    "#4472c4",
    "#a5a5a5",
    "#255e91",
    "#843c0c",
    "#375623",
    "#5f497a",
    "#772c2a",
]


def plot_cvrp_routes(
    inst_name: str,
    coords: np.ndarray,
    routes: list[list[int]],
    objective: float,
    out_path: Path,
    *,
    seed: int | None = None,
    label: str = "CVRP Solution",
) -> Path:
    depot = 0
    n = coords.shape[0] - 1
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.scatter(
        coords[1:, 0],
        coords[1:, 1],
        c="#2e75b6",
        s=55,
        zorder=3,
        edgecolors="white",
        linewidths=0.6,
    )
    for i in range(1, n + 1):
        ax.annotate(
            str(i),
            (coords[i, 0], coords[i, 1]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            color="#1a1a1a",
        )

    ax.scatter(
        coords[depot, 0],
        coords[depot, 1],
        c="#c00000",
        s=140,
        marker="s",
        zorder=5,
        edgecolors="white",
        linewidths=1.0,
    )
    ax.annotate(
        "0",
        (coords[depot, 0], coords[depot, 1]),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=10,
        fontweight="bold",
        color="#c00000",
    )

    legend_handles: list[Line2D] = []
    for k, route in enumerate(routes):
        if not route:
            continue
        color = ROUTE_COLORS[k % len(ROUTE_COLORS)]
        seq = [depot] + route + [depot]
        xs = coords[seq, 0]
        ys = coords[seq, 1]
        ax.plot(xs, ys, "-o", color=color, linewidth=1.8, markersize=4, zorder=2, alpha=0.9)
        legend_handles.append(
            Line2D([0], [0], color=color, marker="o", linestyle="-", label=f"Vehicle {k + 1}")
        )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#c00000",
            markersize=10,
            label="Depot",
        )
    )

    title = f"{inst_name} {label} (objective={objective:.2f}"
    if seed is not None:
        title += f", seed={seed}"
    title += ")"
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.92)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_vrptw_routes(
    inst_name: str,
    coords: np.ndarray,
    routes: list[list[int]],
    objective: float,
    out_path: Path,
    *,
    seed: int | None = None,
    feasible: bool | None = None,
) -> Path:
    label = "VRPTW Solution"
    if feasible is not None:
        label += f", feasible={'Yes' if feasible else 'No'}"
    return plot_cvrp_routes(
        inst_name,
        coords,
        routes,
        objective,
        out_path,
        seed=seed,
        label=label,
    )
