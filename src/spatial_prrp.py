import os
import random
import logging
from typing import Dict, Set, List, Any
from multiprocessing import Pool, cpu_count

from src.prrp_data_loader import load_shapefile
from src.utils import (
    construct_adjacency_list,
    find_connected_components,
    find_boundary_areas,
    parallel_execute,
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
# 1. Gapless Random Seed Selection
# ==============================
def get_gapless_seed(adj_list: Dict[int, Set[int]],
                     available_areas: Set[int],
                     assigned_regions: Set[int]) -> int:
    """
    Selects a gapless seed for region growing, ensuring spatial contiguity.

    For the first region (if no regions have been assigned yet), a random area from
    available_areas is selected. For subsequent regions, the function attempts to pick a
    seed from the neighbors of already assigned areas to maintain spatial contiguity.
    If no such neighbor is available, it falls back to selecting a random area from available_areas.

    Parameters:
        adj_list (Dict[int, Set[int]]): The neighborhood graph represented as an adjacency list.
            Keys are area IDs and values are sets of adjacent area IDs.
        available_areas (Set[int]): Set of unassigned area IDs.
        assigned_regions (Set[int]): Set of area IDs that have already been assigned to regions.

    Returns:
        int: The selected seed area ID.

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

    candidate_seeds = set()
    for area in assigned_regions:
        neighbors = set(adj_list.get(area, set()))
        candidate_seeds.update(neighbors.intersection(available_areas))

    if candidate_seeds:
        seed = random.choice(list(candidate_seeds))
        logger.info(f"Gapless seed selected: {seed}")
        return seed

    seed = random.choice(list(available_areas))
    logger.warning(f"No gapless seed found; selecting random area: {seed}")

    return seed


# ==============================
# 2. Region Growing Phase
# ==============================
def grow_region(adj_list: Dict[int, Set[int]],
                available_areas: Set[int],
                target_cardinality: int,
                max_retries: int = 5) -> Set[int]:
    """
    Grows a spatially contiguous region until the target cardinality is reached.

    The region is grown by:
      1. Selecting an initial seed using gapless seed selection.
      2. Expanding the region by randomly adding unassigned neighbors.
      3. Dynamically updating the frontier of candidate areas.
    If the region cannot be grown to meet the target cardinality (due to a lack of available
    neighboring areas), the growth attempt is restarted with a new seed. After max_retries
    unsuccessful attempts, a RuntimeError is raised.

    Parameters:
        adj_list (Dict[int, Set[int]]): The neighborhood graph represented as an adjacency list.
            Keys are area IDs and values are sets of adjacent area IDs.
        available_areas (Set[int]): Set of unassigned area IDs. This set will be updated by removing
            the areas that become part of the successfully grown region.
        target_cardinality (int): The required number of areas in the region.
        max_retries (int): Maximum number of attempts to grow the region before failing.

    Returns:
        Set[int]: A set of area IDs representing the successfully grown region.

    Raises:
        ValueError: If target_cardinality exceeds the number of available areas.
        RuntimeError: If region growth fails after max_retries attempts.
    """
    if target_cardinality > len(available_areas):
        error_msg = (
            f"Target cardinality ({target_cardinality}) exceeds the number of available areas "
            f"({len(available_areas)})."
        )
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

        frontier = set()
        for area in region:
            neighbors = set(adj_list.get(area, set()))
            frontier.update(neighbors.intersection(temp_available))

        while len(region) < target_cardinality:
            if not frontier:
                logger.debug(
                    "Frontier is empty; unable to expand region further.")
                break

            next_area = random.choice(list(frontier))
            region.add(next_area)
            temp_available.remove(next_area)
            logger.debug(
                f"Added area {next_area} to region. Current region size: {len(region)}.")

            frontier.clear()
            for area in region:
                neighbors = set(adj_list.get(area, set()))
                frontier.update(neighbors.intersection(temp_available))

        if len(region) == target_cardinality:
            available_areas.difference_update(region)
            logger.info(
                f"Successfully grown region with target cardinality {target_cardinality}: {region}")
            return region
        else:
            retries += 1
            logger.warning(
                f"Region growth attempt {retries} failed to reach the target cardinality. Retrying with a new seed."
            )

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
        f"Largest connected component selected with {len(largest_component)} areas.")

    return largest_component


# ==============================
# 4. Region Merging Phase
# ==============================
def merge_disconnected_areas(
    adj_list: Dict[int, Set[int]],
    available_areas: Set[int],
    current_region: Set[int],
    parallelize: bool = False
) -> Set[int]:
    """
    Merges disconnected unassigned areas into the current region to ensure spatial contiguity.

    This function checks whether the unassigned areas (available_areas) are spatially contiguous.
    If they are fragmented into multiple connected components, the largest component is retained as the updated
    available_areas, and all smaller disconnected components are merged into the current_region.

    Parameters:
        adj_list (Dict[int, Set[int]]): The neighborhood graph represented as an adjacency list.
        available_areas (Set[int]): Set of unassigned area IDs.
        current_region (Set[int]): The most recently grown region.
        parallelize (bool, optional): Flag to enable parallel execution if applicable. Defaults to False.

    Returns:
        Set[int]: The updated current_region after merging disconnected areas.

    Raises:
        RuntimeError: If no connected components are found in the available areas.
    """
    if parallelize:
        logger.info(
            "Parallelize flag is set, but sequential execution is used for region merging.")

    # Create a subgraph from available areas, ensuring only edges between available areas are retained.
    sub_adj: Dict[int, List[int]] = {
        area: list(set(adj_list.get(area, set())) & available_areas) for area in available_areas
    }

    # Find connected components in the subgraph using the utility function.
    components: List[Set[int]] = find_connected_components(sub_adj)

    if not components:
        logger.error("No connected components found in available areas.")
        raise RuntimeError("No connected components found in available areas.")

    # Identify the largest connected component.
    largest_component: Set[int] = find_largest_component(components)

    # Merge all smaller disconnected components into the current region.
    merged_areas = set()
    for comp in components:
        if comp != largest_component:
            logger.info(
                f"Merging disconnected component with {len(comp)} areas into the current region: {comp}")
            current_region.update(comp)
            merged_areas.update(comp)

    # Ensure merged areas are removed from available_areas.
    available_areas.difference_update(merged_areas)

    logger.info("Completed merging of disconnected unassigned areas.")

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

        # Randomly remove an area
        area_to_remove = random.choice(list(boundary))
        adjusted_region.remove(area_to_remove)
        excess_count -= 1
        logger.info(
            f"Removed boundary area {area_to_remove}; {excess_count} remaining.")

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
                f"Region split into {len(components)} parts. Keeping largest.")
            excess_count += len(removed)  # Adjust excess count

    return adjusted_region


def split_region(region: Set[int],
                 target_cardinality: int,
                 adj_list: Dict[int, Set[int]]) -> Set[int]:
    """
    Adjusts a region’s size by removing excess areas to meet the target cardinality,
    while ensuring that the region remains spatially contiguous.

    Parameters:
        region (Set[int]): The set of area IDs currently in the region.
        target_cardinality (int): The required number of areas for the region.
        adj_list (Dict[int, Set[int]]): The neighborhood graph represented as an adjacency list.

    Returns:
        Set[int]: The adjusted region that meets the target cardinality.

    Raises:
        ValueError: If the region size is below the target cardinality.
    """
    current_size = len(region)
    if current_size < target_cardinality:
        error_msg = f"Region size ({current_size}) is below the target cardinality ({target_cardinality})."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if current_size == target_cardinality:
        logger.info(
            "Region size matches the target cardinality; no splitting needed.")
        return region

    excess_count = current_size - target_cardinality
    logger.info(f"Splitting region: current size = {current_size}, target = {target_cardinality}, "
                f"excess areas to remove = {excess_count}.")

    adjusted_region = region.copy()

    while excess_count > 0:
        boundary = find_boundary_areas(adjusted_region, adj_list)
        if not boundary:
            logger.error("No boundary areas found; cannot remove further.")
            break

        area_to_remove = random.choice(list(boundary))
        adjusted_region.remove(area_to_remove)
        excess_count -= 1
        logger.info(
            f"Removed boundary area {area_to_remove}; {excess_count} remaining.")

        # Check spatial contiguity after each removal
        sub_adj = {area: list(adj_list.get(area, set()) & adjusted_region)
                   for area in adjusted_region}
        components = find_connected_components(sub_adj)

        if len(components) > 1:
            # Region is fragmented, keep the largest and reassign lost areas
            largest_component = max(components, key=len)
            removed_components = [
                comp for comp in components if comp != largest_component]

            logger.warning(
                f"Region split into {len(components)} parts. Keeping largest with {len(largest_component)} areas.")
            adjusted_region = largest_component

            # Restore removed areas if needed
            while len(adjusted_region) < target_cardinality and removed_components:
                extra_areas = removed_components.pop()
                extra_needed = target_cardinality - len(adjusted_region)

                extra_areas_to_add = list(extra_areas)[:extra_needed]
                adjusted_region.update(extra_areas_to_add)

                logger.info(
                    f"Restored {len(extra_areas_to_add)} areas to maintain target size.")

    logger.info(
        f"Region splitting complete. Final region size is {len(adjusted_region)} areas.")

    return adjusted_region


# ==============================
# 6. PRRP Execution Function
# ==============================


def run_prrp(areas: List[Dict], num_regions: int, cardinalities: List[int]) -> List[Set[int]]:
    """
    Executes the full PRRP algorithm, forming the specified number of regions
    while maintaining spatial contiguity and satisfying cardinality constraints.

    Parameters:
        areas (List[Dict]): List of spatial areas with 'id' and 'geometry' attributes.
        num_regions (int): Number of regions to create.
        cardinalities (List[int]): List of target sizes for each region.

    Returns:
        List[Set[int]]: A list of sets, each containing area IDs forming a valid region.
    """
    if num_regions != len(cardinalities):
        raise ValueError(
            "Number of regions must match the length of the cardinalities list.")

    # Construct adjacency list for spatial relationships.
    adj_list = construct_adjacency_list(areas)
    # Ensure that all neighbor values are sets.
    adj_list = {k: set(v) for k, v in adj_list.items()}
    available_areas = set(adj_list.keys())

    # Sort cardinalities in descending order.
    cardinalities.sort(reverse=True)

    regions = []
    for target_cardinality in cardinalities:
        logger.info(f"Growing region with target size: {target_cardinality}")

        try:
            # Grow the region.
            region = grow_region(adj_list, available_areas, target_cardinality)
            # Only perform merge/split if there remain unassigned areas.
            if available_areas:
                merged_region = merge_disconnected_areas(
                    adj_list, available_areas, region)
                final_region = split_region(
                    merged_region, target_cardinality, adj_list)
            else:
                # If no areas remain unassigned, no merge or split is needed.
                final_region = region
            regions.append(final_region)
            logger.info(f"Region finalized with {len(final_region)} areas.")
        except Exception as e:
            logger.error(f"Failed to generate region: {e}")
            return []  # Return an empty result indicating failure

    return regions


# ==============================
# 7. Parallel Execution of PRRP
# ==============================


def _prrp_worker(seed_value: int,
                 areas: List[Dict[str, Any]],
                 num_regions: int,
                 cardinalities: List[int]) -> List[Set[int]]:
    """
    Worker function for parallel PRRP execution. Sets a unique random seed
    for statistical independence, executes one full PRRP solution, and returns it.

    Parameters:
        seed_value (int): The random seed for this worker.
        areas (List[Dict[str, Any]]): List of spatial areas (each area is a dict with keys such as 'id' and 'geometry').
        num_regions (int): The number of regions to create in this solution.
        cardinalities (List[int]): A list specifying the target cardinality for each region.

    Returns:
        List[Set[int]]: A single PRRP solution, represented as a list of sets where each set contains area IDs for a region.
    """
    random.seed(seed_value)
    logger.info(f"Worker started with seed {seed_value}.")
    solution = run_prrp(areas, num_regions, cardinalities)
    logger.info(f"Worker with seed {seed_value} completed a solution.")

    return solution


def run_parallel_prrp(areas: List[Dict[str, Any]],
                      num_regions: int,
                      cardinalities: List[int],
                      solutions_count: int,
                      num_threads: int = None,
                      use_multiprocessing: bool = True) -> List[List[Set[int]]]:
    """
    Runs multiple independent PRRP solutions in parallel.

    Parameters:
        areas (List[Dict[str, Any]]): List of spatial areas with required attributes (e.g., 'id' and 'geometry').
        num_regions (int): Number of regions to create per solution.
        cardinalities (List[int]): List of target sizes for each region.
        solutions_count (int): Number of independent PRRP solutions to generate.
        num_threads (int, optional): Number of parallel threads/processes to use. If None, it defaults to min(solutions_count, cpu_count()).
        use_multiprocessing (bool, optional): If True, uses multiprocessing; otherwise, executes sequentially.
            Defaults to True.

    Returns:
        List[List[Set[int]]]: A list of PRRP solutions. Each solution is a list of sets (each set represents a region).
    """
    # Determine the number of threads/processes to use.
    if num_threads is None:
        num_threads = min(solutions_count, cpu_count())
    logger.info(
        f"Preparing to generate {solutions_count} PRRP solutions using {num_threads} parallel worker(s).")

    # Generate unique random seeds for each solution.
    seeds = [random.randint(0, 2**31 - 1) for _ in range(solutions_count)]
    logger.info(
        f"Generated {solutions_count} random seeds for PRRP solutions.")

    solutions = []
    if use_multiprocessing:
        logger.info(
            "Starting parallel execution of PRRP solutions using multiprocessing.")
        with Pool(processes=num_threads) as pool:
            # Each worker gets a unique seed, along with the areas, number of regions, and cardinalities.
            worker_args = [(seed, areas, num_regions, cardinalities)
                           for seed in seeds]
            solutions = pool.starmap(_prrp_worker, worker_args)
        logger.info("Parallel execution of PRRP solutions completed.")
    else:
        logger.info(
            "Parallelization disabled; executing PRRP solutions sequentially.")
        for seed in seeds:
            solutions.append(_prrp_worker(
                seed, areas, num_regions, cardinalities))
        logger.info("Sequential execution of PRRP solutions completed.")

    return solutions


# ==============================
# 8. Main Execution Block
# ==============================
if __name__ == "__main__":
    # Load the shapefile data.
    # get the absolute path to the shapefile.
    shapefile_path = os.path.abspath(os.path.join(
        os.getcwd(), 'data/cb_2015_42_tract_500k/cb_2015_42_tract_500k.shp'))
    print(f"Path to shape file : {shapefile_path}")

    sample_areas = load_shapefile(shapefile_path)

    # Define the number of regions to create.
    num_regions = 5

    # Define random target cardinalities for each region such that their sum is equal to the total area points.
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
