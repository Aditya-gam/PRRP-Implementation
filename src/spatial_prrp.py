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
from typing import Dict, Set, List, Any, Optional
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
    Selects a gapless seed for region growing. For the first region, a seed is randomly
    chosen from all available areas. For subsequent regions, the algorithm attempts to choose
    a seed from the set of unassigned areas that are spatial neighbors of any already assigned area.
    This helps ensure that the remaining unassigned areas stay contiguous, which is critical when
    growing regions under a strict cardinality constraint.

    Parameters:
        adj_list (Dict[int, Set[int]]): The spatial adjacency list mapping area IDs to the set of
                                        neighboring area IDs.
        available_areas (Set[int]): The set of area IDs that have not yet been assigned to any region.
        assigned_regions (Set[int]): The set of area IDs that have been assigned to regions.

    Returns:
        int: The ID of the chosen seed area.

    Raises:
        ValueError: If there are no available areas from which to select a seed.
    """
    if not available_areas:
        logger.error("No available areas to select a seed.")
        raise ValueError("No available areas to select a seed.")

    # If no regions have yet been assigned, pick a seed at random.
    if not assigned_regions:
        seed = random.choice(list(available_areas))
        logger.info(f"First seed selected randomly: {seed}")
        return seed

    # Compute candidate seeds as the unassigned neighbors of the already assigned areas.
    candidate_seeds = set()
    for area in assigned_regions:
        # Only consider neighbors that are still available.
        candidate_seeds.update(adj_list.get(area, set()) & available_areas)

    # If there are any candidate seeds from the border, choose one uniformly at random.
    if candidate_seeds:
        seed = random.choice(list(candidate_seeds))
        logger.info(
            f"Gapless seed selected from neighbors: {seed} (Candidates: {candidate_seeds})")
        return seed

    # Otherwise, fall back to the original method:
    heap = []
    for area in available_areas:
        connectivity = sum(1 for nbr in adj_list.get(
            area, set()) if nbr in assigned_regions)
        heapq.heappush(heap, (-connectivity, random.random(), area))
    best_candidate = heapq.heappop(heap)[2]
    logger.info(f"Gapless seed selected using heap fallback: {best_candidate}")

    return best_candidate


# ==============================
# 2. Priority-based Region Growing
# ==============================
def grow_region(adj_list: Dict[int, Set[int]],
                available_areas: Set[int],
                target_cardinality: int,
                max_retries: int = 5,
                initial_seed: Optional[int] = None) -> Set[int]:
    """
    Grows a spatially contiguous region until it reaches the target cardinality.

    Parameters:
      adj_list (Dict[int, Set[int]]): The spatial adjacency graph.
      available_areas (Set[int]): Set of area IDs that have not yet been assigned.
      target_cardinality (int): Desired number of areas in the region.
      max_retries (int, optional): Maximum number of attempts if region growth fails.
      initial_seed (Optional[int], optional): If provided and available, used as the starting seed.

    Returns:
      Set[int]: The grown region.

    Raises:
      ValueError: if target_cardinality exceeds the number of available areas.
      RuntimeError: if a region meeting the target is not grown after max_retries.
    """
    if target_cardinality == len(available_areas):
        logger.info(
            f"Target cardinality {target_cardinality} equals the number of available areas; returning all available areas.")
        return available_areas.copy()

    if target_cardinality > len(available_areas):
        error_msg = (f"Target cardinality ({target_cardinality}) exceeds the number of available areas "
                     f"({len(available_areas)}).")
        logger.error(error_msg)
        raise ValueError(error_msg)

    retries = 0
    while retries < max_retries:
        logger.info(f"Region growing attempt {retries + 1}/{max_retries}")
        temp_available = available_areas.copy()
        # Use the provided seed if available; otherwise, select one randomly.
        if initial_seed is not None and initial_seed in temp_available:
            seed = initial_seed
            logger.info(f"Using provided seed: {seed}")
        else:
            full_areas = set(adj_list.keys())
            assigned_regions = full_areas - temp_available
            try:
                seed = get_gapless_seed(
                    adj_list, temp_available, assigned_regions)
            except ValueError as e:
                logger.error(f"Error selecting seed: {e}")
                raise

        region = {seed}
        temp_available.remove(seed)
        logger.debug(
            f"Started region growing with seed {seed}. Initial region: {region}")

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

            _, _, candidate = heapq.heappop(frontier_heap)
            if candidate not in temp_available:
                continue
            region.add(candidate)
            temp_available.remove(candidate)
            logger.debug(
                f"Added area {candidate} to region. Current region size: {len(region)}.")
            for neighbor in adj_list.get(candidate, set()):
                if neighbor in temp_available:
                    score = len(adj_list.get(neighbor, set()
                                             ).intersection(temp_available))
                    heapq.heappush(
                        frontier_heap, (-score, random.random(), neighbor))

        if len(region) == target_cardinality:
            logger.info(
                f"Successfully grown region with target cardinality {target_cardinality}: {region}")
            return region
        else:
            retries += 1
            logger.warning(
                f"Region growth attempt {retries} failed to reach the target cardinality. Retrying with a new seed.")
            # Reset initial_seed for subsequent retries.
            initial_seed = None

    error_msg = f"Region growth failed after {max_retries} attempts."
    logger.error(error_msg)
    raise RuntimeError(error_msg)


# ==============================
# 3. Find Largest Connected Component
# ==============================
def find_largest_component(connected_components: List[Set[int]]) -> Set[int]:
    """
    Identifies the largest contiguous connected component among the provided components.
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
def merge_disconnected_areas(adj_list: Dict[int, Set[int]],
                             available_areas: Set[int],
                             current_region: Set[int],
                             parallelize: bool = False) -> Set[int]:
    """
    Merges disconnected unassigned areas into the current region using DSU.
    """
    if parallelize:
        logger.info("Parallelize flag set, but sequential DSU merging is used.")

    dsu = DisjointSetUnion()
    for area in available_areas:
        dsu.parent[area] = area

    for area in available_areas:
        for neighbor in adj_list.get(area, set()):
            if neighbor in available_areas:
                dsu.union(area, neighbor)

    components = {}
    for area in available_areas:
        root = dsu.find(area)
        components.setdefault(root, set()).add(area)

    components_list = list(components.values())
    if not components_list:
        logger.error("No connected components found in available areas.")
        raise RuntimeError("No connected components found in available areas.")

    largest_component = find_largest_component(components_list)

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
    Randomly removes boundary areas from a region until the specified excess count is removed,
    ensuring that spatial contiguity is maintained.
    """
    adjusted_region = region.copy()
    while excess_count > 0:
        boundary = find_boundary_areas(
            adjusted_region, {k: list(v) for k, v in adj_list.items()})
        if not boundary:
            logger.error("No boundary areas found; cannot remove further.")
            raise RuntimeError("No boundary areas available for removal.")

        candidate = min(boundary, key=lambda area: len(
            adj_list.get(area, set()).intersection(adjusted_region)))
        adjusted_region.remove(candidate)
        excess_count -= 1
        logger.info(
            f"Removed boundary area {candidate}; {excess_count} removals remaining.")

        sub_adj = {area: list(set(adj_list.get(area, set()))
                              & adjusted_region) for area in adjusted_region}
        components = find_connected_components(sub_adj)

        if len(components) > 1:
            largest_component = max(components, key=len)
            removed = adjusted_region - largest_component
            adjusted_region = largest_component
            logger.warning(
                "Region split into multiple parts. Keeping largest component.")
            excess_count += len(removed)
    return adjusted_region


def split_region(region: Set[int],
                 target_cardinality: int,
                 adj_list: Dict[int, Set[int]]) -> Set[int]:
    """
    Adjusts the region size to meet the target cardinality by removing excess boundary areas.
    Optionally, attempts to restore areas if the region becomes too small.
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
def run_prrp(areas: List[Dict], num_regions: int, cardinalities: List[int],
             max_solution_attempts: int = 10) -> List[Set[int]]:
    """
    Executes the full PRRP algorithm to partition the areas into 'num_regions' regions
    that satisfy the cardinality constraints while maintaining spatial contiguity.

    The algorithm implements gapless seed selection:
      - The first region uses a random seed.
      - For subsequent regions, a "seed queue" is maintained from the unassigned neighbors
        of already grown regions. This helps to keep the unassigned areas contiguous.

    Parameters:
      areas (List[Dict]): List of areas, each with at least "id" and "geometry" keys.
      num_regions (int): Number of regions to form.
      cardinalities (List[int]): Target cardinalities for each region (must sum to the total number of areas).
      max_solution_attempts (int, optional): Maximum attempts to find a valid partition.

    Returns:
      List[Set[int]]: A list of regions (each a set of area IDs).

    Raises:
      ValueError: if num_regions does not match the length of cardinalities.
      RuntimeError: if a valid solution cannot be generated within max_solution_attempts.
    """
    if num_regions != len(cardinalities):
        raise ValueError(
            "Number of regions must match the length of the cardinalities list.")

    from src.utils import construct_adjacency_list  # Ensure proper import

    # Build the spatial adjacency list and ensure neighbor lists are sets.
    adj_list = construct_adjacency_list(areas)
    adj_list = {k: set(v) for k, v in adj_list.items()}
    original_available = set(adj_list.keys())

    # Sort cardinalities in descending order for greater flexibility.
    cardinalities.sort(reverse=True)

    # Initialize a seed queue for gapless seed selection.
    seed_queue: Set[int] = set()

    for attempt in range(max_solution_attempts):
        logger.info(f"Solution attempt {attempt + 1}/{max_solution_attempts}")
        available_areas = original_available.copy()
        regions = []
        seed_queue.clear()

        try:
            # Grow regions one by one.
            for idx, target_cardinality in enumerate(cardinalities):
                logger.info(
                    f"Growing region {idx+1} with target size: {target_cardinality}")
                if idx == 0 or not seed_queue:
                    initial_seed = None
                else:
                    valid_seeds = list(
                        seed_queue.intersection(available_areas))
                    initial_seed = random.choice(
                        valid_seeds) if valid_seeds else None
                    if initial_seed is not None:
                        seed_queue.remove(initial_seed)
                        logger.info(
                            f"Selected seed {initial_seed} from seed queue for region {idx+1}.")

                region = grow_region(
                    adj_list, available_areas, target_cardinality, initial_seed=initial_seed)
                available_areas.difference_update(region)
                logger.info(f"Region {idx+1} grown with {len(region)} areas.")

                # Update seed queue with neighbors of the newly grown region that are still available.
                for area in region:
                    for neighbor in adj_list.get(area, set()):
                        if neighbor in available_areas:
                            seed_queue.add(neighbor)

                regions.append(region)
                logger.info(
                    f"Region {idx+1} finalized with {len(region)} areas.")

            # Validate that all areas are assigned.
            if set().union(*regions) == original_available and len(regions) == num_regions:
                logger.info("Successfully generated a valid solution.")
                return regions
            else:
                raise RuntimeError(
                    "Partitioning incomplete: not all areas were assigned.")
        except Exception as e:
            logger.warning(f"Solution attempt {attempt + 1} failed: {e}")
            continue

    raise RuntimeError(
        "Failed to generate a valid solution after maximum attempts.")


def _prrp_worker(seed_value: int,
                 areas: List[Dict[str, Any]],
                 num_regions: int,
                 cardinalities: List[int]) -> List[Set[int]]:
    """
    Worker function for parallel PRRP execution.
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
