"""
Graph-Based PRRP Implementation

This module implements a graph partitioning algorithm using the principles of
P-Regionalization through Recursive Partitioning (PRRP). It ensures:
    - Connectivity preservation via articulation point checks (using Tarjan’s Algorithm helpers).
    - Efficient handling of graphs through an adjacency list (via construct_adjacency_list).
    - Recursive partitioning with growth, merging of disconnected areas, and splitting of oversized partitions.
    
The graph input is expected to be provided in METIS format (parsed via metis_parser.py)
or already as an adjacency list. This implementation leverages utilities from utils.py.
"""

import logging
import random
from collections import deque
from typing import Dict, Set, List

# Import required functions from utils.
from src.utils import (
    construct_adjacency_list,
    find_articulation_points,    # if needed for extra logging/debugging.
    random_seed_selection,
    find_connected_components,
    find_boundary_areas,
    is_articulation_point
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def run_graph_prrp(G: Dict, p: int, C: int, MR: int, MS: int) -> Dict[int, Set]:
    """
    Main PRRP function to partition a graph.

    Parameters:
        G (Dict): Input graph as an adjacency list (node -> neighbors).
        p (int): Desired number of partitions.
        C (int): Target partition cardinality (ideal number of nodes per partition).
        MR (int): Maximum number of retries for growing a partition.
        MS (int): Maximum allowed partition size before splitting.

    Returns:
        Dict[int, Set]: Mapping of partition IDs to sets of nodes.

    Process:
        1. Validate input (number of nodes must be >= p).
        2. Construct/ensure the adjacency list using construct_adjacency_list().
        3. Initialize an empty partition mapping and unassigned nodes set.
        4. Iteratively:
            - Select a seed node (using gapless seed selection).
            - Grow a partition (using grow_partition).
            - Merge any disconnected areas (using merge_disconnected_areas).
            - If the partition exceeds MS, split it (using split_partition).
            - Remove the partition’s nodes from the unassigned set.
        5. If unassigned nodes remain, assign them to the smallest partition.
    """
    # Ensure proper graph format.
    G_adj = construct_adjacency_list(G)
    all_nodes = set(G_adj.keys())

    if len(all_nodes) < p:
        logger.error(
            "Number of nodes is less than the number of desired partitions.")
        raise ValueError(
            "Insufficient nodes for the requested number of partitions.")

    partitions = {}
    partition_id = 1
    unassigned = set(all_nodes)

    while unassigned and partition_id <= p:
        # Select seed node using the gapless method.
        assigned_nodes = set()
        for part in partitions.values():
            assigned_nodes |= part
        try:
            seed = random_seed_selection(
                G_adj, assigned_nodes, method="gapless")
            if seed not in unassigned:
                seed = random.choice(list(unassigned))
        except ValueError:
            seed = random.choice(list(unassigned))

        # Grow the partition from the selected seed.
        partition = grow_partition(G_adj, unassigned, partition_id, C, MR)
        logger.info(
            f"Grew partition {partition_id} with {len(partition)} nodes.")

        # Ensure the partition is connected by merging any disconnected areas.
        partition = merge_disconnected_areas(G_adj, unassigned, partition)
        logger.info(
            f"After merging, partition {partition_id} has {len(partition)} nodes.")

        # If the partition is too large, split it into smaller connected regions.
        if len(partition) > MS:
            logger.info(
                f"Partition {partition_id} exceeds maximum size {MS}. Splitting...")
            new_parts = split_partition(G_adj, partition, C)
            for np in new_parts:
                partitions[partition_id] = np
                logger.info(
                    f"Created partition {partition_id} with {len(np)} nodes after splitting.")
                partition_id += 1
        else:
            partitions[partition_id] = partition
            partition_id += 1

        # Remove assigned nodes from unassigned.
        unassigned -= partition

    # Assign any remaining unassigned nodes to the smallest partition.
    if unassigned:
        logger.info(
            "Assigning remaining unassigned nodes to the smallest partition.")
        smallest_part = min(partitions.items(),
                            key=lambda item: len(item[1]))[0]
        partitions[smallest_part].update(unassigned)
        unassigned.clear()

    return partitions


def grow_partition(G: Dict, U: Set, p: int, c: int, MR: int) -> Set:
    """
    Grows a partition by expanding from a seed until reaching the target cardinality.

    Parameters:
        G (Dict): Graph as an adjacency list.
        U (Set): Set of unassigned nodes.
        p (int): Identifier of the current partition (used for logging).
        c (int): Target number of nodes for the partition.
        MR (int): Maximum number of retries if growth stalls.

    Returns:
        Set: Set of nodes forming the grown partition.

    Process:
        - Select a seed node (using gapless seed selection).
        - Use an iterative BFS (queue-based) to add adjacent nodes.
        - Prefer adding neighbors that are not articulation points.
        - If no non-articulation candidate exists, add an articulation point.
        - If the queue empties before reaching the target size, try adding a new seed (up to MR retries).
    """
    partition = set()
    attempts = 0

    # Select the initial seed.
    try:
        seed = random_seed_selection(G, set(), method="gapless")
        if seed not in U:
            seed = random.choice(list(U))
    except ValueError:
        seed = random.choice(list(U))

    partition.add(seed)
    U.discard(seed)
    queue = deque([seed])

    while queue and len(partition) < c:
        current = queue.popleft()
        added_any = False
        # First, gather candidates that are NOT articulation points.
        non_articulation_candidates = [
            nbr for nbr in G[current] if nbr in U and not is_articulation_point(G, nbr)]
        if non_articulation_candidates:
            for neighbor in non_articulation_candidates:
                if neighbor in U:
                    partition.add(neighbor)
                    U.discard(neighbor)
                    queue.append(neighbor)
                    added_any = True
                    if len(partition) >= c:
                        break
        else:
            # If no safe candidate exists, fall back to adding any unassigned neighbor.
            for neighbor in G[current]:
                if neighbor in U:
                    partition.add(neighbor)
                    U.discard(neighbor)
                    queue.append(neighbor)
                    added_any = True
                    if len(partition) >= c:
                        break

        # If nothing was added from the current node and the queue is empty, try to add a candidate from adjacent nodes.
        if not added_any and not queue and len(partition) < c and U:
            adjacent_candidates = set()
            for node in partition:
                adjacent_candidates |= (G[node] & U)
            if adjacent_candidates:
                new_seed = random.choice(list(adjacent_candidates))
            else:
                new_seed = random.choice(list(U))
            partition.add(new_seed)
            U.discard(new_seed)
            queue.append(new_seed)
            attempts += 1
            if attempts >= MR:
                logger.warning(
                    f"Partition {p} growth stalled after {MR} retries.")
                break

    return partition


def merge_disconnected_areas(G: Dict, U: Set, Pi: Set) -> Set:
    """
    Merges disconnected subcomponents within a partition to ensure overall connectivity.

    Parameters:
        G (Dict): Graph as an adjacency list.
        U (Set): Unassigned nodes (unused in this function but included for consistency).
        Pi (Set): The current partition (may be disconnected).

    Returns:
        Set: A merged, connected partition.

    Process:
        - Build the induced subgraph for the nodes in Pi.
        - Compute its connected components using DFS.
        - If all connected components are singletons (isolated nodes), return the original partition.
          Otherwise, return the largest connected component.
    """
    # Build the induced subgraph for nodes in Pi.
    induced_adj = {node: {nbr for nbr in G[node] if nbr in Pi} for node in Pi}
    # Prepare input for connected component analysis.
    comp_input = {node: list(neighbors)
                  for node, neighbors in induced_adj.items()}
    components = find_connected_components(comp_input)

    # If all connected components have size 1, return the original partition.
    if all(len(comp) == 1 for comp in components):
        return Pi

    # Otherwise, return only the largest connected component.
    largest_component = max(components, key=len)
    logger.info(
        f"Partition was disconnected; returning largest connected component with {len(largest_component)} nodes out of {len(Pi)}."
    )

    return largest_component


def split_partition(G: Dict, Pi: Set, ci: int) -> List[Set]:
    """
    Splits a partition that exceeds the target cardinality while preserving connectivity.

    Parameters:
        G (Dict): Graph as an adjacency list.
        Pi (Set): The partition to be split.
        ci (int): Target cardinality for each resulting partition.

    Returns:
        List[Set]: List of partitions obtained after splitting.

    Process:
        - If Pi is within the target, return it as a single-element list.
        - Identify boundary nodes (nodes adjacent to nodes outside Pi) using find_boundary_areas.
        - Remove a number of boundary nodes (preferably not articulation points) so that the
          remaining partition is of size ci.
        - Determine connected components of the removed nodes to form additional partitions.
    """
    if len(Pi) <= ci:
        return [Pi]

    removed_nodes = set()
    current_partition = set(Pi)
    excess = len(Pi) - ci
    attempts = 0
    max_attempts = 10 * excess  # Arbitrary maximum attempts.

    while len(removed_nodes) < excess and attempts < max_attempts:
        # Note: find_boundary_areas expects a region and an adjacency list.
        # Here, we build a mini-adjacency list for current_partition.
        mini_adj = {node: list(G[node] & current_partition)
                    for node in current_partition}
        boundary_nodes = find_boundary_areas(current_partition, mini_adj)
        candidates = [
            node for node in boundary_nodes if not is_articulation_point(G, node)]
        if not candidates:
            candidates = list(boundary_nodes)
        if not candidates:
            break
        node_to_remove = random.choice(candidates)
        current_partition.remove(node_to_remove)
        removed_nodes.add(node_to_remove)
        attempts += 1

    # Get connected components from the removed nodes.
    removed_adj = {node: [nbr for nbr in G[node]
                          if nbr in removed_nodes] for node in removed_nodes}
    new_components = find_connected_components(removed_adj)

    partitions = [current_partition]
    partitions.extend(new_components)

    logger.info(
        f"Split partition into {len(partitions)} partitions with target cardinality {ci}.")
    return partitions
