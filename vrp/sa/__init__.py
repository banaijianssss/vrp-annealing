"""模拟退火求解器。"""
from vrp.sa.cvrp import SAResult as CVRPSAResult, solve_cvrp_sa
from vrp.sa.presets import PRESETS, PRESET_CHOICES, SAPreset, normalize_preset
from vrp.sa.vrptw import SAResult as VRPTWSAResult, solve_vrptw_sa

__all__ = [
    "CVRPSAResult",
    "VRPTWSAResult",
    "PRESETS",
    "PRESET_CHOICES",
    "SAPreset",
    "normalize_preset",
    "solve_cvrp_sa",
    "solve_vrptw_sa",
]
