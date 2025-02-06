import random
import logging
from typing import Dict, Set

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

    # If no regions have been assigned yet, return a random seed from available_areas.
    if not assigned_regions:
        seed = random.choice(list(available_areas))
        logger.info(f"First seed selected randomly: {seed}")

        return seed

    # Attempt to pick a seed that is adjacent to any of the already assigned areas.
    candidate_seeds = set()
    for area in assigned_regions:
        neighbors = adj_list.get(area, set())
        candidate_seeds.update(neighbors.intersection(available_areas))

    if candidate_seeds:
        seed = random.choice(list(candidate_seeds))
        logger.info(f"Gapless seed selected: {seed}")

        return seed

    # Last resort: no spatially adjacent candidate found; pick any random area.
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

    # Represent the complete set of area IDs from the adjacency list.
    full_areas = set(adj_list.keys())
    retries = 0

    while retries < max_retries:
        logger.info(f"Region growing attempt {retries + 1}/{max_retries}")

        # Create a temporary copy of available_areas for this growth attempt.
        temp_available = available_areas.copy()
        # Compute assigned_regions as those areas not available.
        assigned_regions = full_areas - temp_available

        try:
            seed = get_gapless_seed(adj_list, temp_available, assigned_regions)
        except ValueError as e:
            logger.error(f"Error selecting seed: {e}")
            raise

        # Initialize the region with the seed and remove it from the temporary available areas.
        region = {seed}
        temp_available.remove(seed)
        logger.debug(f"Started region growing with seed {seed}. Initial region: {region}")

        # Initialize the frontier as all unassigned neighbors of the current region.
        frontier = set()
        for area in region:
            neighbors = adj_list.get(area, set())
            frontier.update(neighbors.intersection(temp_available))

        # Expand the region until the target cardinality is met.
        while len(region) < target_cardinality:
            if not frontier:
                logger.debug("Frontier is empty; unable to expand region further.")
                break  # Unable to grow further; this attempt will be retried.

            # Randomly select the next area from the current frontier.
            next_area = random.choice(list(frontier))
            region.add(next_area)
            temp_available.remove(next_area)
            logger.debug(
                f"Added area {next_area} to region. Current region size: {len(region)}."
            )

            # Update the frontier with unassigned neighbors of the updated region.
            frontier.clear()
            for area in region:
                neighbors = adj_list.get(area, set())
                frontier.update(neighbors.intersection(temp_available))

        if len(region) == target_cardinality:
            # Successful growth: update the available_areas by removing the assigned region.
            available_areas.difference_update(region)
            logger.info(
                f"Successfully grown region with target cardinality {target_cardinality}: {region}"
            )

            return region
        else:
            retries += 1
            logger.warning(
                f"Region growth attempt {retries} failed to reach the target cardinality. "
                f"Retrying with a new seed."
            )

    error_msg = f"Region growth failed after {max_retries} attempts."
    logger.error(error_msg)
    
    raise RuntimeError(error_msg)
