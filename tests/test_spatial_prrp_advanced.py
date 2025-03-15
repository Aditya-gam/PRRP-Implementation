"""
tests/test_spatial_prrp_advanced.py

Advanced Test Suite for the P-Regionalization through Recursive Partitioning (PRRP) Algorithm.

This test suite verifies:
  - Functionality using both synthetic and real-world spatial datasets.
  - Correct behavior of gapless seed selection, region growing, DSU-based merging, and region splitting.
  - Full pipeline execution (both sequential and parallel).
  - Spatial contiguity, cardinality constraints, and complete assignment of areas.
  - Proper error handling for invalid inputs.
  - Basic performance metrics (execution time thresholds).

The real dataset is loaded from:
    data/cb_2015_42_tract_500k/cb_2015_42_tract_500k.shp
Ensure that this file is present relative to the project root.
"""

import os
import time
import unittest
import random
from copy import deepcopy
from typing import Dict, Set, List, Any

# Import PRRP functions from the spatial_prrp module.
from temp.spatial_prrp import (
    get_gapless_seed,
    grow_region,
    merge_disconnected_areas,
    split_region,
    run_prrp,
    run_parallel_prrp
)
# Import utility functions.
from src.utils import find_connected_components, construct_adjacency_list
# Import shapefile loader.
from src.prrp_data_loader import load_shapefile


# -------------------------------
# Helper Functions for Test Data
# -------------------------------
def generate_synthetic_graph() -> Dict[int, Set[int]]:
    """
    Generates a small synthetic adjacency list for testing.

    Returns:
        Dict[int, Set[int]]: A manually defined spatial graph.
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


def generate_synthetic_areas() -> List[Dict[str, Any]]:
    """
    Generates a synthetic list of spatial areas with dummy geometry.

    Returns:
        List[Dict[str, Any]]: A list of area dictionaries.
    """
    return [{'id': i, 'geometry': None} for i in range(1, 13)]


# -------------------------------
# Advanced Test Suite
# -------------------------------
class TestSpatialPRRPAdvanced(unittest.TestCase):
    def setUp(self):
        """
        Initializes both synthetic and real-world datasets for testing.
        """
        # Synthetic dataset.
        self.synthetic_adj = generate_synthetic_graph()
        self.synthetic_available = set(self.synthetic_adj.keys())
        self.synthetic_areas = generate_synthetic_areas()
        self.synthetic_cardinalities = [4, 4, 4]
        self.synthetic_num_regions = len(self.synthetic_cardinalities)

        # Attempt to load the real-world shapefile.
        shapefile_path = os.path.join(
            os.getcwd(), 'data', 'cb_2015_42_tract_500k', 'cb_2015_42_tract_500k.shp')
        try:
            self.real_areas = load_shapefile(shapefile_path)
        except Exception as e:
            self.real_areas = None
            print(f"Warning: Real shapefile could not be loaded: {e}")

    # -------------------------------
    # Synthetic Dataset Tests
    # -------------------------------
    def test_get_gapless_seed_synthetic(self):
        """Verifies that a valid seed is selected from available areas using synthetic data."""
        assigned = {1, 2, 3}
        seed = get_gapless_seed(
            self.synthetic_adj, self.synthetic_available, assigned)
        self.assertIn(seed, self.synthetic_available,
                      "Seed must be from available areas.")

    def test_grow_region_synthetic(self):
        """Tests region growing on synthetic data produces a region of the target size and is contiguous."""
        target = 4
        available_copy = deepcopy(self.synthetic_available)
        region = grow_region(self.synthetic_adj,
                             available_copy, target_cardinality=target)
        self.assertEqual(len(region), target,
                         "Region size should match target.")
        subgraph = {node: list(self.synthetic_adj.get(
            node, set()) & region) for node in region}
        components = find_connected_components(subgraph)
        self.assertEqual(len(components), 1,
                         "Region must be spatially contiguous.")

    def test_merge_disconnected_areas_synthetic(self):
        """
        Tests DSU-based merging on synthetic data.
        Simulates disconnection of area 12 and verifies that it is merged.
        """
        region = {1, 2, 3, 4}
        available = deepcopy(self.synthetic_available) - region
        # Remove neighbors for area 12 to simulate isolation.
        available.discard(8)
        available.discard(11)
        available.add(12)
        merged_region = merge_disconnected_areas(
            self.synthetic_adj, available, region)
        self.assertIn(12, merged_region, "Isolated area should be merged.")

    def test_split_region_synthetic(self):
        """Tests that an oversized synthetic region is properly split to meet the target cardinality."""
        region = {1, 2, 3, 4, 5, 6}
        target = 4
        adjusted = split_region(
            region, target_cardinality=target, adj_list=self.synthetic_adj)
        self.assertEqual(len(adjusted), target,
                         "Region should be trimmed to target size.")
        subgraph = {node: list(self.synthetic_adj.get(
            node, set()) & adjusted) for node in adjusted}
        self.assertEqual(len(find_connected_components(subgraph)), 1,
                         "Trimmed region must remain contiguous.")

    def test_run_prrp_synthetic(self):
        """Tests full PRRP pipeline on synthetic dataset."""
        regions = run_prrp(
            self.synthetic_areas, self.synthetic_num_regions, self.synthetic_cardinalities)
        self.assertEqual(len(regions), self.synthetic_num_regions,
                         "Should produce correct number of regions.")
        for idx, region in enumerate(regions):
            self.assertEqual(len(region), self.synthetic_cardinalities[idx],
                             f"Region {idx+1} must meet its cardinality.")
        union_regions = set().union(*regions)
        expected = set(range(1, 13))
        self.assertEqual(union_regions, expected,
                         "All areas should be assigned.")

    def test_run_parallel_prrp_synthetic(self):
        """Tests parallel execution on synthetic data and checks for statistical independence."""
        sol_count = 3
        solutions = run_parallel_prrp(
            self.synthetic_areas, self.synthetic_num_regions, self.synthetic_cardinalities,
            solutions_count=sol_count, num_threads=2
        )
        self.assertEqual(len(solutions), sol_count,
                         "Should generate the requested number of solutions.")
        expected = set(range(1, 13))
        for sol in solutions:
            self.assertEqual(len(sol), self.synthetic_num_regions,
                             "Each solution must have the correct number of regions.")
            self.assertEqual(set().union(*sol), expected,
                             "All areas should be assigned in each solution.")
        unique = {frozenset(frozenset(region) for region in sol)
                  for sol in solutions}
        self.assertGreater(
            len(unique), 1, "Solutions should be statistically independent.")

    # -------------------------------
    # Real Shapefile Dataset Tests
    # -------------------------------
    def test_load_shapefile_real(self):
        """Verifies that the real shapefile loads a non-empty list of areas."""
        self.assertIsNotNone(
            self.real_areas, "Shapefile must load without error.")
        self.assertGreater(len(self.real_areas), 0,
                           "Shapefile should contain area records.")

    def test_run_prrp_real(self):
        """
        Runs the full PRRP pipeline on the real dataset.
        Sets up target cardinalities based on the number of loaded areas.
        Verifies that:
          - The correct number of regions is produced.
          - Each region's size matches the target.
          - The union of all regions equals the full set of area IDs.
        """
        if self.real_areas is None:
            self.skipTest("Real shapefile dataset not available.")
        total_areas = len(self.real_areas)
        # For testing, we divide the total areas into 5 regions with nearly equal sizes.
        num_regions = 5
        base = total_areas // num_regions
        cardinals = [base] * num_regions
        # Adjust the last region so that the sum equals total_areas.
        cardinals[-1] += total_areas - sum(cardinals)

        regions = run_prrp(self.real_areas, num_regions, cardinals)
        self.assertEqual(len(regions), num_regions,
                         "Should produce the correct number of regions.")
        for idx, region in enumerate(regions):
            self.assertEqual(len(region), cardinals[idx],
                             f"Region {idx+1} should have cardinality {cardinals[idx]}.")
        # Check that union of regions equals all area IDs (assuming IDs are unique).
        union_ids = set().union(*regions)
        # Assume the areas have an 'id' attribute (if not, indices can be used)
        expected_ids = {area['id'] for area in self.real_areas}
        self.assertEqual(union_ids, expected_ids,
                         "All areas must be assigned across regions.")

    def test_regions_contiguity_real(self):
        """
        For the real dataset, verify that each region produced by PRRP is spatially contiguous.
        This uses the find_connected_components utility.
        """
        if self.real_areas is None:
            self.skipTest("Real shapefile dataset not available.")
        total_areas = len(self.real_areas)
        num_regions = 5
        base = total_areas // num_regions
        cardinals = [base] * num_regions
        cardinals[-1] += total_areas - sum(cardinals)
        regions = run_prrp(self.real_areas, num_regions, cardinals)
        # Reconstruct the adjacency list from the real dataset.
        adj_list = construct_adjacency_list(self.real_areas)
        # Convert neighbor lists to sets.
        adj_list = {k: set(v) for k, v in adj_list.items()}
        for idx, region in enumerate(regions):
            sub_adj = {node: list(adj_list.get(node, set()) & region)
                       for node in region}
            comps = find_connected_components(sub_adj)
            self.assertEqual(
                len(comps), 1, f"Region {idx+1} must be spatially contiguous.")

    # -------------------------------
    # Performance Metrics
    # -------------------------------
    def test_region_growing_performance(self):
        """
        Measures the execution time for growing a region on synthetic data.
        Asserts that the operation completes within a specified threshold.
        """
        target = 4
        available_copy = deepcopy(self.synthetic_available)
        start_time = time.time()
        grow_region(self.synthetic_adj, available_copy,
                    target_cardinality=target)
        elapsed = time.time() - start_time
        self.assertLess(
            elapsed, 1.0, "Region growing should complete within 1 second for synthetic data.")

    # -------------------------------
    # Edge Cases and Error Handling
    # -------------------------------
    def test_run_prrp_invalid_inputs(self):
        """
        Verifies that run_prrp raises a ValueError when the number of regions
        does not match the length of the cardinalities list.
        """
        with self.assertRaises(ValueError):
            run_prrp(self.synthetic_areas, num_regions=4,
                     cardinalities=[5, 5, 5])  # 4 != len([5,5,5])

    def test_empty_dataset(self):
        """
        Checks that functions gracefully handle an empty dataset.
        """
        empty_areas = []
        with self.assertRaises(ValueError):
            run_prrp(empty_areas, num_regions=1, cardinalities=[1])


if __name__ == '__main__':
    unittest.main()
