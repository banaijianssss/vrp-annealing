"""结果与参考数据读写。"""

from vrp.io.reference import load_reference_distances
from vrp.io.results import write_solution_txt, write_summary_json

__all__ = [
    "load_reference_distances",
    "write_solution_txt",
    "write_summary_json",
]
