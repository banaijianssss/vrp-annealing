"""VRP 模拟退火求解包（CVRP / VRPTW）。"""
from vrp.sa import solve_cvrp_sa, solve_vrptw_sa

__version__ = "1.0.0"
__all__ = ["solve_cvrp_sa", "solve_vrptw_sa", "__version__"]
