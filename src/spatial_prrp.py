"""
spatial_prrp.py

This module implements the PRRP (P‑Regionalization through Recursive Partitioning)
algorithm for partitioning a set of spatial areas into contiguous regions that exactly
meet given cardinality constraints. The algorithm proceeds in three phases:
    1. Region Growing: A region is grown from a randomly selected seed using a simple
       random-based frontier update while ensuring that the remaining unassigned areas
       contain a contiguous component large enough for the target region.
    2. Region Merging: Any disconnected components among the unassigned areas are merged
       into the current region.
    3. Region Splitting: If the merged region exceeds the target size, excess areas are
       trimmed (and if necessary, areas are restored) so that the region meets the cardinality.
       
A random reference distribution is generated by running the algorithm (potentially in parallel)
with different random seeds.
"""

import os
import random
import logging
import multiprocessing
from multiprocessing import Pool, cpu_count
from typing import List, Set, Tuple

import geopandas as gpd
import networkx as nx
from shapely.strtree import STRtree

# ------------------------------------------------------------------------------
# Configure logging (only major events are logged)
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# 1. Build Spatial Graph from Shapefile
# ------------------------------------------------------------------------------
def create_spatial_network(shp_path: str) -> Tuple[nx.Graph, gpd.GeoDataFrame]:
    """
    Reads a shapefile and constructs a spatial network (as a NetworkX graph).
    Each area (polygon) becomes a node; an edge is added between two nodes if
    their geometries intersect (shared boundaries).

    Parameters:
        shp_path: Path to the shapefile.

    Returns:
        A tuple containing the spatial network (graph) and the GeoDataFrame.
    """
    gdf = gpd.read_file(shp_path)
    net = nx.Graph()
    logger.info(f"Loaded {len(gdf)} areas from the shapefile.")

    # Add nodes for each area (using the index as node id)
    for node_id in gdf.index:
        net.add_node(node_id)

    # Build a spatial index using STRtree for efficient neighbor search.
    spatial_index = STRtree(gdf.geometry)
    geometries = list(gdf.geometry)

    # For each area, query the spatial index to add edges to neighbors.
    for idx, geom in enumerate(geometries):
        possible_idxs = spatial_index.query(geom)
        for jdx in possible_idxs:
            if idx != jdx:
                # Check for intersection (using a threshold if needed)
                if geom.intersects(geometries[jdx]):
                    net.add_edge(gdf.index[idx], gdf.index[jdx])
    logger.info(
        f"Constructed spatial network with {net.number_of_nodes()} nodes and {net.number_of_edges()} edges.")
    return net, gdf


# ------------------------------------------------------------------------------
# 2. Region Growing Phase (Random-based Frontier Expansion)
# ------------------------------------------------------------------------------
def expand_region_randomly(
    net: nx.Graph,
    avail: Set[int],
    target_size: int,
    seed_node: int,
    max_attempts: int = 5
) -> Set[int]:
    """
    Grows a contiguous region from a given seed using a random-based frontier update.
    This modified version respects the provided seed (if still available) to ensure
    the gapless seed selection strategy described in the paper.

    Parameters:
        net: The spatial network (graph) where nodes represent areas.
        avail: Set of currently unassigned node IDs (will be modified locally).
        target_size: The desired number of nodes in the region.
        seed_node: The starting node for the region.
        max_attempts: Maximum attempts (restarts) allowed to reach the target size.

    Returns:
        A set of node IDs that forms a region exactly of size target_size.

    Raises:
        RuntimeError: If no valid region of the target size can be grown after max_attempts.
    """
    for attempt in range(max_attempts):
        temp_avail = avail.copy()
        region_nodes = set()
        # Use the provided seed if it is still available; otherwise, choose a new seed.
        if seed_node not in temp_avail:
            components = list(nx.connected_components(
                net.subgraph(temp_avail)))
            if not components:
                raise RuntimeError(
                    "No available component to select seed from.")
            largest_comp = max(components, key=len)
            seed_node = random.choice(list(largest_comp))
        region_nodes.add(seed_node)
        temp_avail.remove(seed_node)

        # Initialize the frontier with the unassigned neighbors of the seed.
        frontier = set(net.neighbors(seed_node)).intersection(temp_avail)

        # Grow the region until the target size is reached.
        while len(region_nodes) < target_size and frontier:
            candidate = random.choice(list(frontier))
            region_nodes.add(candidate)
            temp_avail.remove(candidate)
            frontier.discard(candidate)
            frontier.update(set(net.neighbors(candidate)
                                ).intersection(temp_avail))

        if len(region_nodes) == target_size:
            avail.intersection_update(temp_avail)
            logger.info(
                f"Region grown to target size {target_size} on attempt {attempt + 1}."
            )

            return region_nodes
        else:
            logger.info(
                f"Region growth attempt {attempt + 1} failed to reach target size {target_size}. Retrying..."
            )
    raise RuntimeError(
        f"Region growth failed: could not grow region of target size {target_size} after {max_attempts} attempts."
    )


# ------------------------------------------------------------------------------
# 3. Region Merging Phase (Merge Disconnected Unassigned Components)
# ------------------------------------------------------------------------------
def integrate_components(
    net: nx.Graph,
    avail: Set[int],
    region_nodes: Set[int]
) -> Set[int]:
    """
    Merges any disconnected unassigned components (from avail) into the current region.
    Specifically, if the unassigned nodes (avail) break into multiple connected components,
    then all but the largest component are merged into the current region.

    Parameters:
        net: The spatial network.
        avail: Set of unassigned node IDs.
        region_nodes: The current region as a set of node IDs.

    Returns:
        The updated region (with merged nodes).
    """
    subgraph = net.subgraph(avail)
    components = list(nx.connected_components(subgraph))
    if len(components) > 1:
        largest_comp = max(components, key=len)
        # Merge every other (smaller) component into the region.
        for comp in components:
            if comp != largest_comp:
                region_nodes.update(comp)
                avail.difference_update(comp)
        logger.info("Merged disconnected components into the region.")
    return region_nodes


# ------------------------------------------------------------------------------
# 4. Region Splitting Phase (Adjust Region Size)
# ------------------------------------------------------------------------------
def adjust_region_size(
    net: nx.Graph,
    region_nodes: Set[int],
    target_size: int,
    avail: Set[int]
) -> Set[int]:
    """
    Adjusts the region to exactly meet the target size. If the region exceeds the target,
    boundary nodes are removed at random (with connectivity re‐assessment). If the region
    becomes too small, adjacent available nodes are added.

    Parameters:
        net: The spatial network.
        region_nodes: The current region as a set of node IDs.
        target_size: Desired region size.
        avail: Set of unassigned node IDs (will be updated if nodes are added).

    Returns:
        A set of node IDs representing the adjusted region.
    """
    # Function to compute boundary nodes (nodes in region with neighbor outside region)
    def boundary_nodes(region: Set[int]) -> Set[int]:
        return {n for n in region if set(net.neighbors(n)) - region}

    # If the region is larger than target, remove boundary nodes.
    while len(region_nodes) > target_size:
        b_nodes = boundary_nodes(region_nodes)
        if not b_nodes:
            break
        rem_node = random.choice(list(b_nodes))
        region_nodes.remove(rem_node)
        # Optionally, add the removed node back to available.
        avail.add(rem_node)
        # Reassess connectivity – keep only the largest connected part.
        comps = list(nx.connected_components(net.subgraph(region_nodes)))
        if comps:
            region_nodes = max(comps, key=len)
    # If the region falls below target, try to add adjacent nodes from avail.
    while len(region_nodes) < target_size:
        candidates = set()
        for n in region_nodes:
            candidates.update(set(net.neighbors(n)).intersection(avail))
        if not candidates:
            break
        add_node = random.choice(list(candidates))
        region_nodes.add(add_node)
        avail.remove(add_node)
    logger.info(
        f"Region size adjusted to {len(region_nodes)} (target was {target_size}).")
    return region_nodes


# ------------------------------------------------------------------------------
# 5. Overall PRRP Execution
# ------------------------------------------------------------------------------
def run_prrp(
    net: nx.Graph,
    cardinality_list: List[int],
    max_region_attempts: int = 20,
    max_restarts: int = 10
) -> List[Set[int]]:
    """
    Executes the full PRRP algorithm to partition the spatial network into regions
    with the specified cardinalities. The cardinality_list must sum to the total number
    of nodes in the graph. If a partitioning attempt fails due to feasibility issues,
    the algorithm restarts the entire region building process up to max_restarts times.

    This version follows the PRRP framework from the paper by first growing a region
    (using a gapless seed selection strategy), then merging disconnected available areas
    (if the remaining unassigned areas are fragmented) and finally adjusting the region
    to exactly meet its target cardinality. If after adjustment the region does not exactly
    meet the target, the iteration is aborted.

    Parameters:
        net: The spatial network (NetworkX graph).
        cardinality_list: List of integers specifying target sizes for each region,
                          assumed to be sorted in descending order.
        max_region_attempts: Maximum attempts for growing each region.
        max_restarts: Maximum number of full algorithm restarts if a solution is not found.

    Returns:
        A list of regions (each a set of node IDs) that partition the entire graph.

    Raises:
        RuntimeError: If a valid partitioning cannot be generated after max_restarts attempts.
    """
    for restart in range(max_restarts):
        try:
            all_nodes = set(net.nodes())
            available_nodes = all_nodes.copy()
            regions = []
            seed_pool = set()  # Candidate seeds from neighbors of previously built regions

            for idx, target in enumerate(cardinality_list):
                # --- Modified Feasibility Check ---
                # Instead of checking the largest connected component,
                # we only require that the total number of available nodes is at least 'target'.
                if len(available_nodes) < target:
                    raise RuntimeError(
                        f"Not enough available nodes for target {target}.")

                # Build mapping for nodes in components that are large enough.
                components = list(nx.connected_components(
                    net.subgraph(available_nodes)))
                comp_dict = {}
                for comp in components:
                    if len(comp) >= target:
                        for node in comp:
                            comp_dict[node] = comp

                # Select a seed: first try to use a seed from the seed_pool.
                candidate_seeds = seed_pool.intersection(available_nodes)
                valid_candidates = {
                    s for s in candidate_seeds if s in comp_dict}
                if valid_candidates:
                    seed = random.choice(list(valid_candidates))
                else:
                    valid_comps = [
                        comp for comp in components if len(comp) >= target]
                    if not valid_comps:
                        raise RuntimeError(
                            f"No contiguous component is large enough for target {target}."
                        )
                    largest_comp = max(valid_comps, key=len)
                    seed = random.choice(list(largest_comp))
                seed_pool.discard(seed)

                logger.info(
                    f"Growing region {idx+1} with target size {target} using seed {seed}."
                )
                region = expand_region_randomly(net, available_nodes, target, seed,
                                                max_attempts=max_region_attempts)
                if not region or len(region) != target:
                    raise RuntimeError(
                        f"Region growth failed for region {idx+1} (target {target})."
                    )
                # Merge any disconnected available components into the region.
                region = integrate_components(net, available_nodes, region)
                # --- Repair Step ---
                # If available_nodes are fragmented, merge all smaller components (all but the largest)
                rem_comps = list(nx.connected_components(
                    net.subgraph(available_nodes)))
                if len(rem_comps) > 1:
                    largest_rem = max(rem_comps, key=len)
                    for comp in rem_comps:
                        if comp != largest_rem:
                            region.update(comp)
                            available_nodes.difference_update(comp)
                    logger.info(
                        f"After repair merge, region {idx+1} size is {len(region)}."
                    )
                # Adjust the region so that it exactly meets the target.
                region = adjust_region_size(
                    net, region, target, available_nodes)
                if len(region) != target:
                    raise RuntimeError(
                        f"After adjustment, region {idx+1} size {len(region)} != target {target}."
                    )
                # Remove the region's nodes from available_nodes.
                available_nodes.difference_update(region)
                regions.append(region)
                logger.info(
                    f"Region {idx+1} finalized with {len(region)} nodes."
                )

                # Update the seed pool with unassigned neighbors of the newly built region.
                for n in region:
                    seed_pool.update(
                        set(net.neighbors(n)).intersection(available_nodes))

            # Final check: all nodes should be assigned.
            if set().union(*regions) != all_nodes:
                raise RuntimeError(
                    "Partitioning incomplete: not all nodes were assigned to a region.")
            logger.info("Successfully generated all regions.")
            return regions

        except RuntimeError as e:
            logger.info(
                f"Restart attempt {restart+1} failed: {e}. Retrying...")
            continue

    raise RuntimeError(
        "Failed to generate a valid partition after maximum restart attempts.")


# ------------------------------------------------------------------------------
# 6. Parallel Execution of PRRP
# ------------------------------------------------------------------------------
def _prrp_worker(
    seed_val: int,
    net: nx.Graph,
    cardinality_list: List[int],
    max_region_attempts: int
) -> List[Set[int]]:
    """
    Worker function to run PRRP with a specific random seed.
    """
    random.seed(seed_val)
    logger.info(f"Worker started with seed {seed_val}.")
    try:
        sol = run_prrp(net, cardinality_list, max_region_attempts)
        logger.info(
            f"Worker with seed {seed_val} successfully generated a solution.")
        return sol
    except Exception as e:
        logger.error(f"Worker with seed {seed_val} failed: {e}")
        return []


def run_parallel_prrp(
    net: nx.Graph,
    cardinality_list: List[int],
    num_solutions: int,
    num_threads: int = None,
    max_region_attempts: int = 5
) -> List[List[Set[int]]]:
    """
    Generates multiple PRRP solutions in parallel.

    Parameters:
        net: The spatial network.
        cardinality_list: List of target region sizes.
        num_solutions: The number of solutions to generate.
        num_threads: Number of worker threads (defaults to min(num_solutions, CPU cores)).
        max_region_attempts: Maximum attempts for growing each region.

    Returns:
        A list of PRRP solutions (each solution is a list of regions).
    """
    if num_threads is None:
        num_threads = min(num_solutions, cpu_count())
    seeds = [random.randint(0, 2**31 - 1) for _ in range(num_solutions)]
    logger.info(
        f"Generating {num_solutions} solutions using {num_threads} threads.")

    with Pool(processes=num_threads) as pool:
        args = [(s, net, cardinality_list, max_region_attempts) for s in seeds]
        results = pool.starmap(_prrp_worker, args)
    # Filter out any failed solutions.
    solutions = [sol for sol in results if sol]
    logger.info(
        f"Parallel execution completed. {len(solutions)} valid solutions generated.")
    return solutions


# ------------------------------------------------------------------------------
# 7. Main Execution Block (for testing)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Example: Set path to a shapefile (modify the path as needed)
    shapefile_path = os.path.abspath(os.path.join(
        os.getcwd(), 'data', 'your_shapefile.shp'))
    logger.info(f"Shapefile path: {shapefile_path}")

    try:
        spatial_graph, gdf = create_spatial_network(shapefile_path)
    except Exception as e:
        logger.error(f"Failed to load shapefile: {e}")
        exit(1)

    total_nodes = spatial_graph.number_of_nodes()
    num_regions = 5

    # For demonstration, generate target cardinalities that sum to total_nodes.
    # Here we generate (num_regions - 1) random values and assign the remainder to the last region.
    random_targets = [random.randint(5, 15) for _ in range(num_regions - 1)]
    last_target = total_nodes - sum(random_targets)
    cardinality_targets = sorted(random_targets + [last_target], reverse=True)
    logger.info(
        f"Cardinality targets (sorted descending): {cardinality_targets}")

    # Run a single PRRP solution.
    try:
        solution = run_prrp(spatial_graph, cardinality_targets)
        logger.info(
            f"Single PRRP solution generated with {len(solution)} regions.")
    except Exception as e:
        logger.error(f"PRRP execution failed: {e}")
        exit(1)

    # Optionally, run multiple solutions in parallel.
    try:
        parallel_sols = run_parallel_prrp(
            spatial_graph, cardinality_targets, num_solutions=3, num_threads=2)
        logger.info(f"Parallel PRRP generated {len(parallel_sols)} solutions.")
    except Exception as e:
        logger.error(f"Parallel PRRP execution failed: {e}")
