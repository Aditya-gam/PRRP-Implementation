import unittest
import random
from typing import Dict, Set, List
from src.spatial_prrp import (
    get_gapless_seed,
    grow_region,
    merge_disconnected_areas,
    split_region,
    run_prrp,
    run_parallel_prrp
)
from src.utils import construct_adjacency_list, find_connected_components

# ==============================
# Test Data Setup
# ==============================


def generate_test_graph() -> Dict[int, Set[int]]:
    """
    Generates a small, manually designed adjacency list representing spatial areas.

    Returns:
        Dict[int, Set[int]]: A simple adjacency list for testing.
    """
    return {
        1: {2, 5}, 2: {1, 3, 6}, 3: {2, 4, 7}, 4: {3, 8},
        5: {1, 6, 9}, 6: {2, 5, 7, 10}, 7: {3, 6, 8, 11}, 8: {4, 7, 12},
        9: {5, 10}, 10: {6, 9, 11}, 11: {7, 10, 12}, 12: {8, 11}
    }


def generate_test_areas() -> List[Dict[str, any]]:
    """
    Generates a synthetic list of spatial areas with dummy geometry.

    Returns:
        List[Dict[str, any]]: A list of spatial area dictionaries.
    """
    return [{'id': i, 'geometry': None} for i in range(1, 13)]

# ==============================
# Test Suite
# ==============================


class TestSpatialPRRP(unittest.TestCase):

    def setUp(self):
        """
        Initializes test data before each test case.
        """
        self.adj_list = generate_test_graph()
        self.available_areas = set(self.adj_list.keys())
        # Target cardinalities for three regions
        self.cardinalities = [4, 4, 4]
        self.num_regions = len(self.cardinalities)
        self.areas = generate_test_areas()

    # ==============================
    # 1. Test get_gapless_seed
    # ==============================
    def test_get_gapless_seed(self):
        """
        Tests whether get_gapless_seed selects a valid seed while maintaining spatial contiguity.
        """
        assigned_regions = {1, 2, 3}  # Some regions already assigned
        seed = get_gapless_seed(
            self.adj_list, self.available_areas, assigned_regions)
        self.assertIn(seed, self.available_areas,
                      "Seed must be from available areas.")

    # ==============================
    # 2. Test grow_region
    # ==============================
    def test_grow_region(self):
        """
        Tests the grow_region function to ensure it generates a spatially contiguous region.
        """
        target_size = 4
        region = grow_region(self.adj_list, self.available_areas, target_size)
        self.assertEqual(len(region), target_size,
                         "Region must match target cardinality.")

        # Ensure the region is spatially contiguous
        subgraph = {area: self.adj_list[area] & region for area in region}
        components = find_connected_components(subgraph)
        self.assertEqual(len(components), 1, "Region must be contiguous.")

    # ==============================
    # 3. Test merge_disconnected_areas
    # ==============================
    def test_merge_disconnected_areas(self):
        """
        Ensures that all unassigned areas remain spatially contiguous after merging.
        """
        region = {1, 2, 3, 4}
        disconnected_area = {12}  # Intentionally disconnected from region
        self.available_areas.remove(12)

        merged_region = merge_disconnected_areas(
            self.adj_list, self.available_areas, region)
        self.assertIn(12, merged_region, "Disconnected area should be merged.")

    # ==============================
    # 4. Test split_region
    # ==============================
    def test_split_region(self):
        """
        Ensures excess areas are removed while maintaining spatial contiguity.
        """
        region = {1, 2, 3, 4, 5, 6}  # Initial oversized region
        target_size = 4

        adjusted_region = split_region(region, target_size, self.adj_list)
        self.assertEqual(len(adjusted_region), target_size,
                         "Region size must match target cardinality.")

        # Ensure spatial contiguity
        subgraph = {area: self.adj_list[area] &
                    adjusted_region for area in adjusted_region}
        components = find_connected_components(subgraph)
        self.assertEqual(len(components), 1,
                         "Adjusted region must be contiguous.")

    # ==============================
    # 5. Test run_prrp (Full PRRP Execution)
    # ==============================
    def test_run_prrp(self):
        """
        Tests the full PRRP pipeline.
        """
        regions = run_prrp(self.areas, self.num_regions, self.cardinalities)
        self.assertEqual(len(regions), self.num_regions,
                         "Should generate the correct number of regions.")

        for i, region in enumerate(regions):
            self.assertEqual(len(
                region), self.cardinalities[i], f"Region {i} must match cardinality constraint.")

    # ==============================
    # 6. Test run_parallel_prrp
    # ==============================
    def test_run_parallel_prrp(self):
        """
        Tests parallel execution of PRRP.
        """
        solutions_count = 3
        parallel_solutions = run_parallel_prrp(
            self.areas, self.num_regions, self.cardinalities, solutions_count, num_threads=2
        )

        self.assertEqual(len(parallel_solutions), solutions_count,
                         "Parallel execution should generate multiple solutions.")

        for solution in parallel_solutions:
            self.assertEqual(len(solution), self.num_regions,
                             "Each parallel solution must have the correct number of regions.")

    # ==============================
    # 7. Edge Cases
    # ==============================
    def test_grow_region_insufficient_areas(self):
        """
        Ensures that growing a region fails when there aren't enough available areas.
        """
        self.available_areas = {1, 2}  # Insufficient areas
        with self.assertRaises(ValueError):
            grow_region(self.adj_list, self.available_areas,
                        target_cardinality=5)

    def test_run_prrp_invalid_cardinalities(self):
        """
        Tests if run_prrp raises an error when the number of regions does not match cardinalities.
        """
        with self.assertRaises(ValueError):
            run_prrp(self.areas, num_regions=4, cardinalities=[
                     3, 3, 3])  # Mismatch in lengths


# Run tests
if __name__ == '__main__':
    unittest.main()
