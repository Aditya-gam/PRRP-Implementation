"""
spatial_prrp.py

This module implements the P-Regionalization through Recursive Partitioning (PRRP) algorithm,
which grows spatially contiguous regions with given cardinality constraints and merges or splits
regions as needed to maintain spatial contiguity. It also supports parallel execution.
"""

import os
import random
import logging
import heapq
from typing import Dict, Set, List, Any
from multiprocessing import Pool, cpu_count

from src.prrp_data_loader import load_shapefile
from src.utils import (
    construct_adjacency_list,
    find_connected_components,
    find_boundary_areas,
    parallel_execute,
    DisjointSetUnion,  # DSU for region merging
)

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


# ==============================
# 1. Optimized Gapless Seed Selection
# ==============================
def get_gapless_seed(adj_list: Dict[int, Set[int]],
                     available_areas: Set[int],
                     assigned_regions: Set[int]) -> int:
    """
    Selects a gapless seed for region growing by using a heap-based approach.
    For the first region, it randomly picks an area from available_areas.
    For subsequent regions, it computes a connectivity score for each unassigned
    area (number of neighbors in assigned_regions) and uses a min-heap (with negative scores)
    to pick the area with the highest connectivity.

    Parameters:
        adj_list (Dict[int, Set[int]]): Adjacency list of spatial areas.
        available_areas (Set[int]): Set of unassigned area IDs.
        assigned_regions (Set[int]): Set of already assigned area IDs.

    Returns:
        int: Selected seed area ID.

    Raises:
        ValueError: If available_areas is empty.
    """
    if not available_areas:
        logger.error("No available areas to select a seed.")
        raise ValueError("No available areas to select a seed.")

    if not assigned_regions:
        seed = random.choice(list(available_areas))
        logger.info(f"First seed selected randomly: {seed}")
        return seed

    # Build a heap of (negative connectivity, random tie-breaker, area)
    heap = []
    for area in available_areas:
        connectivity = sum(1 for nbr in adj_list.get(
            area, []) if nbr in assigned_regions)
        # Negative score because we want the area with highest connectivity
        heapq.heappush(heap, (-connectivity, random.random(), area))
    best_candidate = heapq.heappop(heap)[2]
    logger.info(f"Gapless seed selected using heap: {best_candidate}")

    return best_candidate


# ==============================
# 2. Priority-based Region Growing
# ==============================
def grow_region(adj_list: Dict[int, Set[int]],
                available_areas: Set[int],
                target_cardinality: int,
                max_retries: int = 5) -> Set[int]:
    """
    Grows a spatially contiguous region to reach the target cardinality.

    Instead of randomly selecting neighbors, this version uses a priority queue to
    always select the neighbor with the highest number of unassigned neighbors,
    thereby reducing retries and fragmentation.

    Parameters:
        adj_list (Dict[int, Set[int]]): Adjacency list of spatial areas.
        available_areas (Set[int]): Unassigned area IDs (this set will NOT be modified in this function).
        target_cardinality (int): The required region size.
        max_retries (int): Maximum allowed retries for growing the region.

    Returns:
        Set[int]: Set of area IDs forming the region.

    Raises:
        ValueError: If target_cardinality exceeds available areas.
        RuntimeError: If region growth fails after max_retries.
    """
    if target_cardinality > len(available_areas):
        error_msg = (f"Target cardinality ({target_cardinality}) exceeds the number of available areas "
                     f"({len(available_areas)}).")
        logger.error(error_msg)
        raise ValueError(error_msg)

    full_areas = set(adj_list.keys())
    retries = 0

    while retries < max_retries:
        logger.info(f"Region growing attempt {retries + 1}/{max_retries}")
        temp_available = available_areas.copy()
        assigned_regions = full_areas - temp_available

        try:
            seed = get_gapless_seed(adj_list, temp_available, assigned_regions)
        except ValueError as e:
            logger.error(f"Error selecting seed: {e}")
            raise

        region = {seed}
        temp_available.remove(seed)
        logger.debug(
            f"Started region growing with seed {seed}. Initial region: {region}")

        # Initialize a priority queue (heap) for expansion candidates.
        frontier_heap = []
        for neighbor in adj_list.get(seed, set()):
            if neighbor in temp_available:
                score = len(adj_list.get(neighbor, set()
                                         ).intersection(temp_available))
                heapq.heappush(
                    frontier_heap, (-score, random.random(), neighbor))

        while len(region) < target_cardinality:
            if not frontier_heap:
                logger.debug(
                    "Frontier is empty; unable to expand region further.")
                break

            # Pop candidate with highest connectivity
            _, _, candidate = heapq.heappop(frontier_heap)
            if candidate not in temp_available:
                continue  # Skip stale entries
            region.add(candidate)
            temp_available.remove(candidate)
            logger.debug(
                f"Added area {candidate} to region. Current region size: {len(region)}.")

            # Add new neighbors from the candidate
            for neighbor in adj_list.get(candidate, set()):
                if neighbor in temp_available:
                    score = len(adj_list.get(neighbor, set()
                                             ).intersection(temp_available))
                    heapq.heappush(
                        frontier_heap, (-score, random.random(), neighbor))

        if len(region) == target_cardinality:
            logger.info(
                f"Successfully grown region with target cardinality {target_cardinality}: {region}")
            # NOTE: Do not update available_areas here; let the caller (run_prrp) handle it.
            return region
        else:
            retries += 1
            logger.warning(
                f"Region growth attempt {retries} failed to reach the target cardinality. Retrying with a new seed.")

    error_msg = f"Region growth failed after {max_retries} attempts."
    logger.error(error_msg)
    raise RuntimeError(error_msg)


# ==============================
# 3. Find Largest Connected Component
# ==============================
def find_largest_component(connected_components: List[Set[int]]) -> Set[int]:
    """
    Identifies the largest contiguous connected component among the provided components.

    Parameters:
        connected_components (List[Set[int]]): A list of sets, where each set represents a connected component of area IDs.

    Returns:
        Set[int]: The largest connected component (by number of areas).

    Raises:
        ValueError: If no connected components are provided.
    """
    if not connected_components:
        logger.error("No connected components found.")
        raise ValueError("No connected components found.")

    largest_component = max(connected_components, key=len)
    logger.info(
        f"Largest connected component has {len(largest_component)} areas.")

    return largest_component


# ==============================
# 4. Optimized Region Merging Phase using Union-Find (DSU)
# ==============================
def merge_disconnected_areas(
    adj_list: Dict[int, Set[int]],
    available_areas: Set[int],
    current_region: Set[int],
    parallelize: bool = False
) -> Set[int]:
    """
    Merges disconnected unassigned areas into the current region using Union-Find (DSU)
    to efficiently identify connected components.

    Parameters:
        adj_list (Dict[int, Set[int]]): Neighborhood graph.
        available_areas (Set[int]): Unassigned area IDs.
        current_region (Set[int]): Current region to be merged with disconnected areas.
        parallelize (bool): Flag for parallel execution (not implemented here).

    Returns:
        Set[int]: Updated current_region after merging.
    """
    if parallelize:
        logger.info("Parallelize flag set, but sequential DSU merging is used.")

    # Build DSU for available areas.
    dsu = DisjointSetUnion()
    for area in available_areas:
        dsu.parent[area] = area

    for area in available_areas:
        for neighbor in adj_list.get(area, set()):
            if neighbor in available_areas:
                dsu.union(area, neighbor)

    # Group areas by their root.
    components = {}
    for area in available_areas:
        root = dsu.find(area)
        components.setdefault(root, set()).add(area)

    components_list = list(components.values())
    if not components_list:
        logger.error("No connected components found in available areas.")
        raise RuntimeError("No connected components found in available areas.")

    largest_component = find_largest_component(components_list)

    # Merge all smaller components into current_region.
    merged_areas = set()
    for comp in components_list:
        if comp != largest_component:
            logger.info(
                f"Merging disconnected component with {len(comp)} areas into the current region: {comp}")
            current_region.update(comp)
            merged_areas.update(comp)

    available_areas.difference_update(merged_areas)
    logger.info("Completed merging of disconnected areas using DSU.")
    if len(current_region) == len(available_areas):
        logger.info(
            "All available areas have been merged into the current region.")
    else:
        logger.warning(
            f"Current region size: {len(current_region)}; Available areas remaining: {len(available_areas)}.")

    return current_region


# ==============================
# 5. Region Splitting Phase
# ==============================
def remove_boundary_areas(region: Set[int],
                          excess_count: int,
                          adj_list: Dict[int, Set[int]]) -> Set[int]:
    """
    Randomly removes boundary areas from a region until the specified excess count
    is removed, while ensuring that spatial contiguity is maintained.

    The function computes the set of boundary areas (areas that have at least one
    neighbor outside the region) and randomly removes one area at a time. After each
    removal, the connectivity of the updated region is checked. If the region splits
    into multiple connected components, only the largest component is retained.

    Parameters:
        region (Set[int]): The current set of area IDs in the region.
        excess_count (int): The number of areas to remove from the region.
        adj_list (Dict[int, Set[int]]): The adjacency list representing spatial neighbors.

    Returns:
        Set[int]: The updated region after removing the excess boundary areas.

    Raises:
        RuntimeError: If no boundary areas can be found to remove when needed.
    """
    # Work on a copy so as not to modify the input region directly.
    adjusted_region = region.copy()
    while excess_count > 0:
        # Identify boundary areas
        boundary = find_boundary_areas(
            adjusted_region, {k: list(v) for k, v in adj_list.items()})
        if not boundary:
            logger.error("No boundary areas found; cannot remove further.")
            raise RuntimeError("No boundary areas available for removal.")

        # Select the boundary area with the least internal connectivity
        candidate = min(boundary, key=lambda area: len(
            adj_list.get(area, set()).intersection(adjusted_region)))
        adjusted_region.remove(candidate)
        excess_count -= 1
        logger.info(
            f"Removed boundary area {candidate}; {excess_count} removals remaining.")

        # Check spatial contiguity
        sub_adj = {area: list(set(adj_list.get(area, set()))
                              & adjusted_region) for area in adjusted_region}
        components = find_connected_components(sub_adj)

        if len(components) > 1:
            # If fragmentation occurs, keep the largest component
            largest_component = max(components, key=len)
            removed = adjusted_region - largest_component
            adjusted_region = largest_component
            logger.warning(
                "Region split into multiple parts. Keeping largest component.")
            excess_count += len(removed)  # Adjust excess count

    return adjusted_region


def split_region(region: Set[int],
                 target_cardinality: int,
                 adj_list: Dict[int, Set[int]]) -> Set[int]:
    """
    Adjusts the region size to meet target cardinality by removing least-connected boundary areas.
    Prioritizes removal based on the number of internal connections (least connected first).
    Ensures the final region is contiguous.

    Parameters:
        region (Set[int]): Current region.
        target_cardinality (int): Desired number of areas.
        adj_list (Dict[int, Set[int]]): Neighborhood graph.

    Returns:
        Set[int]: Adjusted region meeting target cardinality.

    Raises:
        ValueError: If region size is below target.
    """
    current_size = len(region)
    if current_size < target_cardinality:
        error_msg = f"Region size ({current_size}) below target ({target_cardinality})."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if current_size == target_cardinality:
        logger.info("Region size equals target; no splitting required.")
        return region

    excess_count = current_size - target_cardinality
    logger.info(
        f"Splitting region: current size = {current_size}, target = {target_cardinality}, removing {excess_count} areas.")

    adjusted_region = remove_boundary_areas(region, excess_count, adj_list)
    removed_areas = set()

    # Attempt to restore areas if region is too small.
    if len(adjusted_region) < target_cardinality:
        needed_count = target_cardinality - len(adjusted_region)
        logger.warning(
            f"Region too small after splitting. Attempting to restore {needed_count} areas.")

        removed_areas = region - adjusted_region
        for area in list(removed_areas):
            if needed_count <= 0:
                break

            if any(neighbor in adjusted_region for neighbor in adj_list.get(area, [])):
                adjusted_region.add(area)
                removed_areas.remove(area)
                needed_count -= 1
                logger.info(
                    f"Restored area {area}; {needed_count} areas needed.")
                if needed_count == 0:
                    break

    # Final connectivity check: if the region is fragmented, keep the largest connected component.
    final_sub_adj = {area: list(adj_list.get(
        area, set()) & adjusted_region) for area in adjusted_region}
    final_components = find_connected_components(final_sub_adj)

    if len(final_components) > 1:
        final_region = max(final_components, key=len)
        logger.warning(
            f"Final region not contiguous; using largest component with {len(final_region)} areas.")
        return final_region

    logger.info(
        f"Region splitting complete. Final region size: {len(adjusted_region)}.")
    if len(adjusted_region) == target_cardinality:
        logger.info("Region size matches target cardinality.")
    else:
        logger.warning(
            f"Final region size {len(adjusted_region)} does not match target {target_cardinality}.")

    return adjusted_region


# ==============================
# 6. PRRP Execution and Parallel Runs
# ==============================
def run_prrp(areas: List[Dict], num_regions: int, cardinalities: List[int]) -> List[Set[int]]:
    """
    Executes the full PRRP algorithm to form the specified number of regions while maintaining
    spatial contiguity and satisfying cardinality constraints.

    Parameters:
        areas (List[Dict]): List of spatial areas (each with 'id' and 'geometry').
        num_regions (int): Number of regions to create.
        cardinalities (List[int]): List of target sizes for each region.

    Returns:
        List[Set[int]]: List of regions (each a set of area IDs).

    This implementation ensures that the available areas are updated only once per region.
    The growing step (grow_region) now returns a region without modifying available_areas.
    The run_prrp function then removes the finalized region from the available areas.
    """
    if num_regions != len(cardinalities):
        raise ValueError(
            "Number of regions must match the length of the cardinalities list.")

    # Construct the spatial adjacency list.
    adj_list = construct_adjacency_list(areas)
    # Ensure all neighbor lists are sets.
    adj_list = {k: set(v) for k, v in adj_list.items()}
    available_areas = set(adj_list.keys())

    # Sort cardinalities in descending order.
    cardinalities.sort(reverse=True)
    regions = []

    for target_cardinality in cardinalities:
        logger.info(f"Growing region with target size: {target_cardinality}")
        try:
            # Grow the region; note that grow_region no longer updates available_areas.
            region = grow_region(adj_list, available_areas, target_cardinality)
            # Remove the grown region from available_areas immediately.
            available_areas.difference_update(region)
            # If there are any available areas left, adjust the region for connectivity.
            if available_areas:
                merged_region = merge_disconnected_areas(
                    adj_list, available_areas.copy(), region)
                final_region = split_region(
                    merged_region, target_cardinality, adj_list)
            else:
                final_region = region
            # Update the global available areas based on the final region.
            available_areas.difference_update(final_region)
            regions.append(final_region)
            logger.info(f"Region finalized with {len(final_region)} areas.")
        except Exception as e:
            logger.error(f"Failed to generate region: {e}")
            return []  # Return an empty result indicating failure

    return regions


def _prrp_worker(seed_value: int,
                 areas: List[Dict[str, Any]],
                 num_regions: int,
                 cardinalities: List[int]) -> List[Set[int]]:
    """
    Worker function for parallel PRRP execution. Sets a unique random seed
    for statistical independence, executes one full PRRP solution, and returns it.

    Parameters:
        seed_value (int): The random seed for this worker.
        areas (List[Dict[str, Any]]): List of spatial areas.
        num_regions (int): The number of regions to create.
        cardinalities (List[int]): Target sizes for regions.

    Returns:
        List[Set[int]]: A single PRRP solution.
    """
    random.seed(seed_value)
    logger.info(f"Worker started with seed {seed_value}.")
    solution = run_prrp(areas, num_regions, cardinalities)
    logger.info(f"Worker with seed {seed_value} completed solution.")
    if not solution:
        logger.error(
            f"Worker with seed {seed_value} failed to generate a valid solution.")
    else:
        logger.info(
            f"Worker with seed {seed_value} generated a valid solution.")
    return solution


def run_parallel_prrp(areas: List[Dict[str, Any]],
                      num_regions: int,
                      cardinalities: List[int],
                      solutions_count: int,
                      num_threads: int = None,
                      use_multiprocessing: bool = True) -> List[List[Set[int]]]:
    """
    Executes multiple PRRP solutions in parallel.

    Parameters:
        areas (List[Dict[str, Any]]): Spatial areas.
        num_regions (int): Regions to create per solution.
        cardinalities (List[int]): Target sizes for regions.
        solutions_count (int): Number of solutions to generate.
        num_threads (int): Number of parallel threads/processes.
        use_multiprocessing (bool): Whether to use multiprocessing.

    Returns:
        List[List[Set[int]]]: List of PRRP solutions.
    """
    if num_threads is None:
        num_threads = min(solutions_count, cpu_count())
    logger.info(
        f"Generating {solutions_count} PRRP solutions using {num_threads} workers.")

    seeds = [random.randint(0, 2**31 - 1) for _ in range(solutions_count)]
    logger.info(f"Generated seeds: {seeds}")

    solutions = []
    if use_multiprocessing:
        logger.info("Starting parallel execution using multiprocessing.")
        with Pool(processes=num_threads) as pool:
            worker_args = [(seed, areas, num_regions, cardinalities)
                           for seed in seeds]
            solutions = pool.starmap(_prrp_worker, worker_args)
        logger.info("Parallel PRRP execution completed.")
    else:
        logger.info("Executing PRRP solutions sequentially.")
        for seed in seeds:
            solutions.append(_prrp_worker(
                seed, areas, num_regions, cardinalities))
        logger.info("Sequential PRRP execution completed.")

    return solutions


# ==============================
# 8. Main Execution Block (for testing)
# ==============================
if __name__ == "__main__":
    # Get absolute path to the shapefile.
    shapefile_path = os.path.abspath(os.path.join(
        os.getcwd(), 'data/cb_2015_42_tract_500k/cb_2015_42_tract_500k.shp'))
    print(f"Path to shape file: {shapefile_path}")

    sample_areas = load_shapefile(shapefile_path)
    if sample_areas is None:
        logger.error("Failed to load areas from shapefile.")
        exit(1)

    num_regions = 5
    total_areas = len(sample_areas)
    cardinalities = [random.randint(5, 15) for _ in range(num_regions - 1)]
    cardinalities.append(total_areas - sum(cardinalities))

    logger.info("Running a single PRRP solution...")
    single_solution = run_prrp(sample_areas, num_regions, cardinalities)
    logger.info(f"Single PRRP solution: {single_solution}")

    logger.info("Running parallel PRRP solutions...")
    parallel_solutions = run_parallel_prrp(
        sample_areas, num_regions, cardinalities, solutions_count=3, num_threads=2, use_multiprocessing=True)
    logger.info(f"Generated parallel PRRP solutions: {parallel_solutions}")
