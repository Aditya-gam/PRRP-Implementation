"""
spatial_partitioning.py

This module implements a revised version of the P‐Regionalization through Recursive Partitioning (PRRP)
algorithm. It partitions spatial areas into contiguous regions that exactly satisfy given cardinality constraints.
The algorithm operates in three phases:
  1. Region Growing (with feasibility checking and a simple random-based frontier update)
  2. Region Merging (to ensure remaining unassigned areas remain contiguous)
  3. Region Splitting (to trim overgrown regions to the exact target size)

This implementation supports parallel execution and is designed to generate a random reference distribution
of sample solutions for statistical inference.
"""

import os
import random
import logging
from typing import Dict, Set, List, Any, Optional
from multiprocessing import Pool, cpu_count

# Assume similar functionality as before
from src.data_loader import load_shapefile
from src.utils import (
    construct_adjacency_list,
    find_connected_components,
    find_boundary_areas,
)

# Configure a module-level logger (use less verbose logging for production)
logger = logging.getLogger("spatial_partitioning")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(fmt)
    logger.addHandler(handler)


# ------------------------------------------------------------
# 1. Gapless Seed Selection (renamed)
# ------------------------------------------------------------
def select_seed(adj_map: Dict[int, Set[int]],
                free_areas: Set[int],
                assigned: Set[int]) -> int:
    """
    Selects a seed area for region growth. For the first region, a seed is chosen randomly from free_areas.
    For subsequent regions, a seed is chosen from the free neighbors of already assigned areas.
    """
    if not free_areas:
        logger.error("No free areas available for seed selection.")
        raise ValueError("No free areas available for seed selection.")

    if not assigned:
        chosen = random.choice(list(free_areas))
        logger.info(f"Initial seed selected randomly: {chosen}")
        return chosen

    candidate_set = set()
    for a in assigned:
        candidate_set.update(adj_map.get(a, set()) & free_areas)

    if candidate_set:
        chosen = random.choice(list(candidate_set))
        logger.info(f"Seed selected from free neighbors: {chosen}")
        return chosen

    # Fallback to random selection
    chosen = random.choice(list(free_areas))
    logger.info(f"Fallback seed selected randomly: {chosen}")
    return chosen


# ------------------------------------------------------------
# 2. Simplified Region Growing Phase
# ------------------------------------------------------------
def grow_single_region(adj_map: Dict[int, Set[int]],
                       free_areas: Set[int],
                       target_size: int,
                       max_attempts: int = 5,
                       initial_seed: Optional[int] = None) -> Set[int]:
    """
    Grows a contiguous region from a seed until it reaches target_size.
    Uses a simple random-based frontier update.
    Performs a feasibility check before growth.
    """
    if target_size > len(free_areas):
        error_text = f"Target size ({target_size}) exceeds available areas ({len(free_areas)})."
        logger.error(error_text)
        raise ValueError(error_text)

    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        logger.info(f"Region growth attempt {attempt} of {max_attempts}.")
        current_free = free_areas.copy()

        # Determine seed
        if initial_seed is not None and initial_seed in current_free:
            seed_area = initial_seed
            logger.info(f"Using provided seed: {seed_area}")
        else:
            seed_area = select_seed(adj_map, current_free, set(
                adj_map.keys()) - current_free)
        region_set = {seed_area}
        current_free.remove(seed_area)

        # Feasibility check: ensure the largest connected component among current_free is >= target_size
        comp_list = find_connected_components({area: list(adj_map.get(area, set()) & current_free)
                                               for area in current_free})
        if comp_list and max(len(comp) for comp in comp_list) < target_size - len(region_set):
            logger.warning(
                "Feasibility check failed for region growth; retrying with a new seed.")
            continue

        # Initialize frontier with free neighbors of the seed
        frontier = set(adj_map.get(seed_area, set())) & current_free

        while len(region_set) < target_size and frontier:
            candidate = random.choice(list(frontier))
            region_set.add(candidate)
            current_free.remove(candidate)
            # Update frontier: remove the chosen candidate and add its free neighbors
            frontier = (
                frontier - {candidate}) | (adj_map.get(candidate, set()) & current_free)

        if len(region_set) == target_size:
            logger.info(f"Region successfully grown with {target_size} areas.")
            # Remove the region areas from the global free set
            free_areas.difference_update(region_set)
            return region_set
        else:
            logger.warning("Region growth did not meet target; retrying.")
    raise RuntimeError(
        f"Failed to grow a region of size {target_size} after {max_attempts} attempts.")


# ------------------------------------------------------------
# 3. Simplified Region Merging Phase
# ------------------------------------------------------------
def merge_unassigned_components(adj_map: Dict[int, Set[int]],
                                free_areas: Set[int],
                                region_set: Set[int]) -> Set[int]:
    """
    Merges any disconnected free (unassigned) areas that are adjacent to the current region.
    Uses the connected components from the free areas’ subgraph.
    """
    # Build a temporary sub-adjacency list for free_areas
    temp_adj = {area: list(adj_map.get(area, set()) & free_areas)
                for area in free_areas}
    components = find_connected_components(temp_adj)
    if not components:
        return region_set

    largest_comp = max(components, key=len)
    # Merge other components (if they are adjacent to the region)
    for comp in components:
        if comp is not largest_comp:
            if any(adj_map.get(r, set()) & comp for r in region_set):
                region_set.update(comp)
                free_areas.difference_update(comp)
    logger.info("Merging phase completed.")
    return region_set


# ------------------------------------------------------------
# 4. Simplified Region Splitting Phase
# ------------------------------------------------------------
def adjust_region_size(region_set: Set[int],
                       target_size: int,
                       adj_map: Dict[int, Set[int]],
                       free_areas: Set[int]) -> Set[int]:
    """
    Trims the region if its size exceeds target_size by removing boundary areas.
    Uses a simple loop to remove random boundary nodes until region_set has the target size.
    """
    while len(region_set) > target_size:
        boundary_nodes = find_boundary_areas(
            region_set, {k: list(v) for k, v in adj_map.items()})
        if not boundary_nodes:
            logger.error("No boundary nodes available for trimming.")
            break
        removal = random.choice(list(boundary_nodes))
        region_set.remove(removal)
        free_areas.add(removal)
        # Ensure contiguity: if region_set splits, keep only the largest connected component.
        sub_adj = {area: list(adj_map.get(area, set()) & region_set)
                   for area in region_set}
        comps = find_connected_components(sub_adj)
        if len(comps) > 1:
            region_set = max(comps, key=len)
        logger.info(f"Trimmed region size to {len(region_set)}.")
    return region_set


# ------------------------------------------------------------
# 5. Full PRRP Execution (Building the Partition)
# ------------------------------------------------------------
def build_spatial_partitions(area_list: List[Dict],
                             region_count: int,
                             size_targets: List[int],
                             max_solution_attempts: int = 10) -> List[Set[int]]:
    """
    Partitions the input areas into region_count regions that satisfy the exact cardinality constraints (size_targets).
    Uses the three-phase approach (growing, merging, splitting) to generate a valid partition.
    A feasibility check is performed before growing each region.
    """
    if region_count != len(size_targets):
        raise ValueError(
            "The number of regions must match the number of size targets provided.")

    # Build the adjacency map using the provided utility function.
    adjacency_map = construct_adjacency_list(area_list)
    # Ensure neighbor lists are sets.
    adjacency_map = {key: set(val) for key, val in adjacency_map.items()}
    global_free = set(adjacency_map.keys())

    # Sort size targets in descending order (largest regions first)
    sorted_targets = sorted(size_targets, reverse=True)
    partition_regions: List[Set[int]] = []
    seed_collection: Set[int] = set()

    for attempt in range(1, max_solution_attempts + 1):
        logger.info(
            f"Full partitioning attempt {attempt} of {max_solution_attempts}.")
        current_free = global_free.copy()
        partition_regions.clear()
        seed_collection.clear()

        try:
            for idx, target in enumerate(sorted_targets):
                logger.info(
                    f"Growing region {idx + 1} with target size {target}.")
                # Feasibility check: ensure the largest connected component in current_free is large enough.
                temp_adj = {area: list(adjacency_map.get(
                    area, set()) & current_free) for area in current_free}
                comps = find_connected_components(temp_adj)
                if not comps or max(len(comp) for comp in comps) < target:
                    raise RuntimeError(
                        f"Not enough contiguous areas to form region with target size {target}.")

                # For regions after the first, try to pick a seed from the seed_collection.
                if idx > 0 and seed_collection:
                    valid_seed = list(seed_collection & current_free)
                    seed_val = random.choice(
                        valid_seed) if valid_seed else None
                else:
                    seed_val = None

                # Grow the region using the simplified random-based frontier update.
                region_candidate = grow_single_region(
                    adjacency_map, current_free, target, initial_seed=seed_val)

                # Merge any disconnected free areas that are adjacent.
                region_candidate = merge_unassigned_components(
                    adjacency_map, current_free, region_candidate)

                # If region_candidate exceeds the target size, trim it.
                if len(region_candidate) > target:
                    region_candidate = adjust_region_size(
                        region_candidate, target, adjacency_map, current_free)

                logger.info(
                    f"Region {idx + 1} finalized with {len(region_candidate)} areas.")
                partition_regions.append(region_candidate)

                # Update seed_collection with neighbors of the current region.
                for area in region_candidate:
                    seed_collection.update(
                        adjacency_map.get(area, set()) & current_free)

            # Final feasibility check: all areas should be assigned.
            if set().union(*partition_regions) == global_free and len(partition_regions) == region_count:
                logger.info("Successfully generated a valid partition.")
                return partition_regions
            else:
                raise RuntimeError(
                    "Partitioning incomplete: some areas remain unassigned.")
        except Exception as e:
            logger.warning(f"Attempt {attempt} failed: {e}")
            continue

    raise RuntimeError(
        "Failed to generate a valid partition after maximum attempts.")


# ------------------------------------------------------------
# 6. Parallel Execution Support
# ------------------------------------------------------------
def worker_run_partitioning(seed: int,
                            area_list: List[Dict[str, Any]],
                            region_count: int,
                            size_targets: List[int]) -> List[Set[int]]:
    """
    Worker function for parallel execution of partitioning.
    """
    random.seed(seed)
    logger.info(f"Worker started with seed {seed}.")
    partition = build_spatial_partitions(area_list, region_count, size_targets)
    logger.info(f"Worker with seed {seed} completed partitioning.")
    return partition


def run_parallel_partitioning(area_list: List[Dict[str, Any]],
                              region_count: int,
                              size_targets: List[int],
                              num_solutions: int,
                              num_workers: Optional[int] = None,
                              use_mp: bool = True) -> List[List[Set[int]]]:
    """
    Runs the partitioning algorithm in parallel to generate multiple solutions.
    """
    if num_workers is None:
        num_workers = min(num_solutions, cpu_count())
    logger.info(
        f"Generating {num_solutions} solutions using {num_workers} worker(s).")
    seeds = [random.randint(0, 2**31 - 1) for _ in range(num_solutions)]
    logger.info(f"Generated seeds: {seeds}")

    results = []
    if use_mp:
        with Pool(processes=num_workers) as pool:
            worker_args = [(s, area_list, region_count, size_targets)
                           for s in seeds]
            results = pool.starmap(worker_run_partitioning, worker_args)
    else:
        for s in seeds:
            results.append(worker_run_partitioning(
                s, area_list, region_count, size_targets))
    logger.info("Parallel partitioning execution completed.")
    return results


# ------------------------------------------------------------
# 7. Main Execution Block (for testing)
# ------------------------------------------------------------
if __name__ == "__main__":
    shp_path = os.path.abspath(os.path.join(
        os.getcwd(), 'data/cb_2015_42_tract_500k/cb_2015_42_tract_500k.shp'))
    logger.info(f"Using shapefile at: {shp_path}")
    areas_data = load_shapefile(shp_path)
    if areas_data is None:
        logger.error("Failed to load areas from shapefile.")
        exit(1)

    num_regions_req = 5
    total_count = len(areas_data)
    # Generate random size targets that sum to total_count.
    size_targets_list = [random.randint(5, 15)
                         for _ in range(num_regions_req - 1)]
    size_targets_list.append(total_count - sum(size_targets_list))
    logger.info(f"Size targets for regions: {size_targets_list}")

    logger.info("Running a single partitioning solution...")
    partition_solution = build_spatial_partitions(
        areas_data, num_regions_req, size_targets_list)
    logger.info(f"Single partitioning solution: {partition_solution}")

    logger.info("Running parallel partitioning solutions...")
    parallel_solutions = run_parallel_partitioning(areas_data, num_regions_req, size_targets_list,
                                                   num_solutions=3, num_workers=2, use_mp=True)
    logger.info(f"Parallel partitioning solutions: {parallel_solutions}")
