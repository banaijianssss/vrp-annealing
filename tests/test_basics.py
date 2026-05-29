"""基础模块测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vrp.constants import discover_instance_files
from vrp.data.instance import load_instance, load_vrptw_instance
from vrp.io.reference import gap_pct, load_reference_distances
from vrp.paths import INSTANCE_DIR, ROOT
from vrp.sa.cvrp_heuristics import is_feasible as cvrp_is_feasible
from vrp.sa.metrics import route_distance, route_load, solution_distance as metric_solution_distance
from vrp.util.run_output import make_run_output_dir


class TestInstanceIO(unittest.TestCase):
    def test_load_cvrp_and_vrptw(self) -> None:
        files = discover_instance_files(INSTANCE_DIR)
        self.assertGreaterEqual(len(files), 1)
        path = INSTANCE_DIR / files[0]
        cvrp = load_instance(path)
        vrptw = load_vrptw_instance(path)
        self.assertEqual(cvrp.name, vrptw.name)
        self.assertEqual(cvrp.n_customers, vrptw.n_customers)
        dist = cvrp.dist_matrix()
        self.assertEqual(dist.shape, (cvrp.n_customers + 1, cvrp.n_customers + 1))

    def test_instances_sorted_by_customer_count(self) -> None:
        files = discover_instance_files(INSTANCE_DIR)
        counts = [load_instance(INSTANCE_DIR / f).n_customers for f in files]
        self.assertEqual(counts, sorted(counts))


class TestMetrics(unittest.TestCase):
    def test_route_metrics(self) -> None:
        inst = load_instance(INSTANCE_DIR / discover_instance_files(INSTANCE_DIR)[0])
        dist = inst.dist_matrix()
        route = [1, 2, 3]
        load = route_load(inst.demands, route)
        self.assertGreater(load, 0)
        rd = route_distance(dist, inst.depot, route)
        self.assertGreater(rd, 0)
        total = metric_solution_distance(dist, inst.depot, [route])
        self.assertAlmostEqual(total, rd)


class TestFeasibility(unittest.TestCase):
    def test_cvrp_feasibility_rules(self) -> None:
        inst = load_instance(INSTANCE_DIR / discover_instance_files(INSTANCE_DIR)[0])
        # 未覆盖全部客户 → 不可行
        self.assertFalse(cvrp_is_feasible(inst, [[1]]))
        # 全覆盖的单客户路线集合
        all_singletons = [[i] for i in range(1, inst.n_customers + 1)]
        if len(all_singletons) <= inst.num_vehicles:
            self.assertTrue(cvrp_is_feasible(inst, all_singletons))


class TestRunOutput(unittest.TestCase):
    def test_run_dir_naming(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            d1 = make_run_output_dir(base, seed=42, preset="fast")
            d2 = make_run_output_dir(base, seed=42, preset="fast")
            self.assertTrue(d1.name.startswith("20"))
            self.assertIn("seed42", d1.name)
            self.assertIn("fast", d1.name)
            self.assertTrue((d1 / "plots").is_dir())
            self.assertNotEqual(d1.name, d2.name)


class TestReference(unittest.TestCase):
    def test_gap_pct(self) -> None:
        self.assertAlmostEqual(gap_pct(110.0, 100.0), 10.0)
        self.assertIsNone(gap_pct(110.0, None))

    def test_load_reference_file(self) -> None:
        refs = load_reference_distances()
        self.assertIsInstance(refs, dict)


if __name__ == "__main__":
    unittest.main()
