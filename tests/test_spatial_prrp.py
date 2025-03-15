"""
tests/test_spatial_prrp.py

Unit tests for the P-Regionalization through Recursive Partitioning (PRRP) Algorithm.
This test suite covers:
  - Gapless seed selection (get_gapless_seed)
  - Region growing (grow_region)
  - Region merging (merge_disconnected_areas)
  - Region splitting (split_region)
  - Full PRRP execution (run_prrp)
  - Parallel execution of PRRP (run_parallel_prrp)
  - Loading and processing of a real spatial dataset
  - Construction and PRRP execution on synthetic grid data

Each test ensures that spatial contiguity and cardinality constraints are maintained.
"""

import os
import unittest
import random
from copy import deepcopy
from typing import Dict, Set, List, Any, Tuple

# Import functions to be tested from the PRRP module.
from src.spatial_prrp import (
    get_gapless_seed,
    grow_region,
    merge_disconnected_areas,
    split_region,
    run_prrp,
    run_parallel_prrp
)
# Import utility functions.
from src.utils import find_connected_components, construct_adjacency_list
# Import the shapefile loader from the data loader module.
from src.prrp_data_loader import load_shapefile

# For generating synthetic geometries.
from shapely.geometry import box
import geopandas as gpd


# ==============================
# Helper Functions for Test Data
# ==============================
def generate_test_graph() -> Dict[int, Set[int]]:
    """
    Generates a small, manually designed adjacency list representing spatial areas.
    """
    return {
        1: {2, 5},
        2: {1, 3, 6},
        3: {2, 4, 7},
        4: {3, 8},
        5: {1, 6, 9},
        6: {2, 5, 7, 10},
        7: {3, 6, 8, 11},
        8: {4, 7, 12},
        9: {5, 10},
        10: {6, 9, 11},
        11: {7, 10, 12},
        12: {8, 11}
    }


def generate_test_areas() -> List[Dict[str, Any]]:
    """
    Generates a synthetic list of spatial areas with dummy geometry (None).
    """
    return [{'id': i, 'geometry': None} for i in range(1, 13)]


def generate_grid_areas(rows: int, cols: int) -> List[Dict[str, Any]]:
    """
    Generates a grid of square polygons representing spatial areas.
    Each area is a square of unit length with sequential ids.
    """
    areas = []
    for i in range(rows):
        for j in range(cols):
            area_id = i * cols + j + 1
            polygon = box(j, i, j + 1, i + 1)
            areas.append({'id': area_id, 'geometry': polygon})
    return areas


def generate_grid_test_data(rows: int, cols: int) -> Tuple[List[Dict[str, Any]], Dict[int, Set[int]]]:
    """
    Generates synthetic grid test data and computes the expected rook-adjacency list.
    """
    areas = generate_grid_areas(rows, cols)
    expected_adj_list = {}
    for i in range(rows):
        for j in range(cols):
            area_id = i * cols + j + 1
            neighbors = set()
            if i > 0:
                neighbors.add((i - 1) * cols + j + 1)
            if i < rows - 1:
                neighbors.add((i + 1) * cols + j + 1)
            if j > 0:
                neighbors.add(i * cols + (j - 1) + 1)
            if j < cols - 1:
                neighbors.add(i * cols + (j + 1) + 1)
            expected_adj_list[area_id] = neighbors
    return areas, expected_adj_list


def sample_real_dataset(areas: List[Dict[str, Any]], sample_size: int = 100) -> List[Dict[str, Any]]:
    """
    Returns a representative subset of the dataset.
    If the dataset is larger than 'sample_size', a random sample of that size is returned;
    otherwise, the entire dataset is returned.
    """
    if len(areas) > sample_size:
        return random.sample(areas, sample_size)
    else:
        return areas


# ==============================
# Test Suite
# ==============================
class TestSpatialPRRP(unittest.TestCase):

    def setUp(self):
        """
        Initializes test data before each test case.
        Creates a synthetic spatial graph and a dummy list of areas.
        """
        self.adj_list: Dict[int, Set[int]] = generate_test_graph()
        self.available_areas: Set[int] = set(self.adj_list.keys())
        self.areas: List[Dict[str, Any]] = generate_test_areas()
        self.cardinalities = [4, 4, 4]
        self.num_regions = len(self.cardinalities)

    # ==============================
    # 1. Test get_gapless_seed
    # ==============================
    def test_get_gapless_seed(self):
        assigned_regions = {1, 2, 3}
        seed = get_gapless_seed(
            self.adj_list, self.available_areas, assigned_regions)
        self.assertIn(seed, self.available_areas)
        seed2 = get_gapless_seed(self.adj_list, self.available_areas, set())
        self.assertIn(seed2, self.available_areas)
        with self.assertRaises(ValueError):
            get_gapless_seed(self.adj_list, set(), assigned_regions)

    # ==============================
    # 2. Test grow_region
    # ==============================
    def test_grow_region(self):
        target_size = 4
        available = deepcopy(self.available_areas)
        region = grow_region(self.adj_list, available, target_size)
        self.assertEqual(len(region), target_size)
        subgraph = {area: list(self.adj_list.get(
            area, set()) & region) for area in region}
        components = find_connected_components(subgraph)
        self.assertEqual(len(components), 1)

    def test_grow_region_insufficient_areas(self):
        available = {1, 2}
        with self.assertRaises(ValueError):
            grow_region(self.adj_list, available, 5)

    # ==============================
    # 3. Test merge_disconnected_areas
    # ==============================
    def test_merge_disconnected_areas(self):
        region = {1, 2, 3, 4}
        available = deepcopy(self.available_areas) - region
        available.discard(8)
        available.discard(11)
        available.add(12)
        merged_region = merge_disconnected_areas(
            self.adj_list, available, region)
        self.assertIn(12, merged_region)

    # ==============================
    # 4. Test split_region
    # ==============================
    def test_split_region(self):
        region = {1, 2, 3, 4, 5, 6}
        target_size = 4
        adjusted_region = split_region(region, target_size, self.adj_list)
        self.assertEqual(len(adjusted_region), target_size)
        subgraph = {area: list(self.adj_list.get(
            area, set()) & adjusted_region) for area in adjusted_region}
        components = find_connected_components(subgraph)
        self.assertEqual(len(components), 1)

    # ==============================
    # 5. Test run_prrp (Full PRRP Execution) on Synthetic Data
    # ==============================
    def test_run_prrp(self):
        regions = run_prrp(self.areas, self.num_regions, self.cardinalities)
        self.assertEqual(len(regions), self.num_regions)
        for i, region in enumerate(regions):
            self.assertEqual(len(region), self.cardinalities[i])
        all_region_areas = set().union(*regions)
        expected_areas = set(range(1, 13))
        self.assertEqual(all_region_areas, expected_areas)

    # ==============================
    # 6. Test run_parallel_prrp (Parallel Execution) on Synthetic Data
    # ==============================
    def test_run_parallel_prrp(self):
        solutions_count = 3
        parallel_solutions = run_parallel_prrp(
            self.areas, self.num_regions, self.cardinalities, solutions_count=solutions_count, num_threads=2)
        self.assertEqual(len(parallel_solutions), solutions_count)
        expected_areas = set(range(1, 13))
        for solution in parallel_solutions:
            self.assertEqual(len(solution), self.num_regions)
            union_areas = set().union(*solution)
            self.assertEqual(union_areas, expected_areas)
        unique_solutions = {frozenset(frozenset(region)
                                      for region in sol) for sol in parallel_solutions}
        self.assertGreater(len(unique_solutions), 1)

    # ==============================
    # 7. Test run_prrp_invalid_cardinalities
    # ==============================
    def test_run_prrp_invalid_cardinalities(self):
        with self.assertRaises(ValueError):
            run_prrp(self.areas, num_regions=4, cardinalities=[5, 5, 5])

    # ==============================
    # New Tests: Real Dataset Integration (Using Subset)
    # ==============================
    def test_load_shapefile(self):
        shapefile_path = os.path.abspath(os.path.join(
            os.getcwd(), 'data/cb_2015_42_tract_500k/cb_2015_42_tract_500k.shp'))
        if not os.path.exists(shapefile_path):
            self.skipTest(f"Shapefile not found at {shapefile_path}")
        areas = load_shapefile(shapefile_path)
        # Use a subset to speed up tests if dataset is very large.
        areas = areas if len(areas) < 100 else random.sample(areas, 100)
        self.assertIsNotNone(areas)
        self.assertGreater(len(areas), 0)
        for area in areas:
            self.assertIn('id', area)
            self.assertIn('geometry', area)
            self.assertIsNotNone(area['geometry'])

    def test_run_prrp_real_dataset(self):
        shapefile_path = os.path.abspath(os.path.join(
            os.getcwd(), 'data/cb_2015_42_tract_500k/cb_2015_42_tract_500k.shp'))
        if not os.path.exists(shapefile_path):
            self.skipTest(f"Shapefile not found at {shapefile_path}")
        areas = load_shapefile(shapefile_path)
        # Use a representative subset to keep test run times short.
        areas = areas if len(areas) < 100 else random.sample(areas, 50)
        total_areas = len(areas)
        num_regions = 5
        cardinalities = [random.randint(5, 15) for _ in range(num_regions - 1)]
        cardinalities.append(total_areas - sum(cardinalities))
        regions = run_prrp(areas, num_regions, cardinalities)
        self.assertEqual(len(regions), num_regions)
        all_assigned = set().union(*regions)
        expected_ids = {area['id'] for area in areas}
        self.assertEqual(all_assigned, expected_ids)

    def test_run_parallel_prrp_real_dataset(self):
        shapefile_path = os.path.abspath(os.path.join(
            os.getcwd(), 'data/cb_2015_42_tract_500k/cb_2015_42_tract_500k.shp'))
        if not os.path.exists(shapefile_path):
            self.skipTest(f"Shapefile not found at {shapefile_path}")
        areas = load_shapefile(shapefile_path)
        # Use a representative subset to keep test run times short.
        areas = areas if len(areas) < 100 else random.sample(areas, 50)
        total_areas = len(areas)
        num_regions = 5
        cardinalities = [random.randint(5, 15) for _ in range(num_regions - 1)]
        cardinalities.append(total_areas - sum(cardinalities))
        solutions_count = 3
        parallel_solutions = run_parallel_prrp(
            areas, num_regions, cardinalities, solutions_count=solutions_count, num_threads=2)
        self.assertEqual(len(parallel_solutions), solutions_count)
        expected_ids = {area['id'] for area in areas}
        for solution in parallel_solutions:
            self.assertEqual(len(solution), num_regions)
            union_ids = set().union(*solution)
            self.assertEqual(union_ids, expected_ids)


if __name__ == '__main__':
    unittest.main()
