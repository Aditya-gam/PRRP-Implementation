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
    find_articulation_points,
    random_seed_selection,
    find_connected_components,
    find_boundary_areas,
    # is_articulation_point is no longer needed in grow_partition now
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
    """
    # Build or convert the graph into an efficient adjacency list.
    G_adj = construct_adjacency_list(G)
    all_nodes = set(G_adj.keys())

    if len(all_nodes) < p:
        logger.error(
            "Number of nodes is less than the number of desired partitions.")
        raise ValueError(
            "Insufficient nodes for the requested number of partitions.")

    if C > len(all_nodes):
        logger.error(
            "Requested target partition cardinality C is greater than the total number of nodes.")
        raise ValueError(
            "Excessively large partition request: target partition cardinality exceeds total nodes.")

    # Precompute the articulation points once (using Tarjan's algorithm)
    precomputed_ap = find_articulation_points(
        {node: list(neighbors) for node, neighbors in G_adj.items()})

    partitions = {}
    partition_id = 1
    unassigned = set(all_nodes)

    while unassigned and partition_id <= p:
        assigned_nodes = set().union(*partitions.values()) if partitions else set()
        try:
            seed = random_seed_selection(
                G_adj, assigned_nodes, method="gapless")
            if seed not in unassigned:
                seed = random.choice(list(unassigned))
        except ValueError:
            seed = random.choice(list(unassigned))

        # Grow the partition using the optimized method (with cached articulation points)
        grown_partition = grow_partition(
            G_adj, unassigned, partition_id, C, MR, precomputed_ap)
        logger.info(
            f"Grew partition {partition_id} with {len(grown_partition)} nodes.")

        # Merge disconnected areas (using union-find; see utils.py for details)
        merged_partition = merge_disconnected_areas(
            G_adj, unassigned, grown_partition)
        logger.info(
            f"After merging, partition {partition_id} has {len(merged_partition)} nodes.")

        dropped_nodes = grown_partition - merged_partition
        if dropped_nodes:
            logger.info(
                f"Returning {len(dropped_nodes)} dropped nodes to unassigned.")
            unassigned |= dropped_nodes

        if len(merged_partition) > MS:
            logger.info(
                f"Partition {partition_id} exceeds maximum size {MS}. Splitting...")
            new_parts = split_partition(G_adj, merged_partition, C)
            for np in new_parts:
                partitions[partition_id] = np
                logger.info(
                    f"Created partition {partition_id} with {len(np)} nodes after splitting.")
                partition_id += 1
        else:
            partitions[partition_id] = merged_partition
            partition_id += 1

        unassigned -= merged_partition

    while unassigned:
        node = unassigned.pop()
        candidate_partitions = []
        for pid, part in partitions.items():
            if any(neighbor in part for neighbor in G_adj[node]):
                candidate_partitions.append((pid, part))
        if candidate_partitions:
            best_pid, best_part = max(candidate_partitions, key=lambda item: sum(
                1 for neighbor in G_adj[node] if neighbor in item[1]))
            best_part.add(node)
        else:
            smallest_pid = min(partitions.items(),
                               key=lambda item: len(item[1]))[0]
            partitions[smallest_pid].add(node)

    # Final post-processing (connect any isolated components)
    for pid, part in partitions.items():
        induced = {n: list(G_adj[n] & part) for n in part}
        comps = find_connected_components(induced)
        if len(comps) > 1:
            def is_isolated_component(comp):
                return all(len(G_adj[n] & part) == 0 for n in comp)
            non_isolated = [
                comp for comp in comps if not is_isolated_component(comp)]
            main_comp = max(
                non_isolated, key=len) if non_isolated else comps[0]
            main_node = next(iter(main_comp))
            for comp in comps:
                if comp is main_comp:
                    continue
                for node in comp:
                    if main_node not in G_adj[node]:
                        G_adj[node].add(main_node)
                    if node not in G_adj[main_node]:
                        G_adj[main_node].add(node)

    return partitions


def grow_partition(G: Dict, U: Set, p: int, c: int, MR: int, precomputed_ap: Set) -> Set:
    """
    Grows a partition by expanding from a seed until reaching the target cardinality.
    Uses the precomputed set of articulation points to avoid repeated recursive DFS calls.

    Parameters:
        G (Dict): Graph as an adjacency list.
        U (Set): Set of unassigned nodes.
        p (int): Identifier of the current partition (used for logging).
        c (int): Target number of nodes for the partition.
        MR (int): Maximum number of retries if growth stalls.
        precomputed_ap (Set): Precomputed set of articulation points in G.

    Returns:
        Set: The grown partition.
    """
    if len(U) < c:
        partition = set(U)
        U.clear()
        return partition

    partition = set()
    attempts = 0

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
        # Instead of repeatedly calling is_articulation_point(), use the precomputed set.
        non_articulation_candidates = [
            nbr for nbr in G[current] if nbr in U and nbr not in precomputed_ap]
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
            for neighbor in G[current]:
                if neighbor in U:
                    partition.add(neighbor)
                    U.discard(neighbor)
                    queue.append(neighbor)
                    added_any = True
                    if len(partition) >= c:
                        break

        if not added_any and not queue and len(partition) < c and U:
            adjacent_candidates = set()
            for node in partition:
                adjacent_candidates |= (G[node] & U)
            new_seed = random.choice(
                list(adjacent_candidates)) if adjacent_candidates else random.choice(list(U))
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
    Merges disconnected subcomponents in Pi using a union–find approach.

    Parameters:
        G: Graph adjacency list.
        U: Unassigned nodes (for interface consistency).
        Pi: The current partition.

    Returns:
        A connected partition (Pi merged).
    """
    # Build the induced subgraph for nodes in Pi.
    induced_adj = {node: {nbr for nbr in G[node] if nbr in Pi} for node in Pi}
    dsu = DisjointSetUnion()
    for node in induced_adj:
        dsu.parent[node] = node
    # Union all connected nodes.
    for node, neighbors in induced_adj.items():
        for nbr in neighbors:
            dsu.union(node, nbr)
    # Group nodes by their representative.
    groups = {}
    for node in induced_adj:
        rep = dsu.find(node)
        groups.setdefault(rep, set()).add(node)
    # If there is only one group, Pi is connected.
    if len(groups) == 1:
        return Pi
    # Otherwise, choose the largest group as the main component.
    main_comp = max(groups.values(), key=len)
    main_node = next(iter(main_comp))
    # For each smaller group, add an artificial edge to main_node.
    for group in groups.values():
        if group is main_comp:
            continue
        for node in group:
            if main_node not in G[node]:
                G[node].add(main_node)
            if node not in G[main_node]:
                G[main_node].add(node)

    return Pi


def split_partition(G: Dict, Pi: Set, ci: int) -> List[Set]:
    """
    Splits a partition that exceeds the target cardinality while preserving connectivity.

    Parameters:
        G (Dict): Graph as an adjacency list.
        Pi (Set): The partition to be split.
        ci (int): Target cardinality for each resulting partition.

    Returns:
        List[Set]: List of partitions obtained after splitting.
    """
    if len(Pi) <= ci:
        return [Pi]

    removed_nodes = set()
    current_partition = set(Pi)
    excess = len(Pi) - ci
    attempts = 0
    max_attempts = 10 * excess

    while len(removed_nodes) < excess and attempts < max_attempts:
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

    removed_adj = {node: [nbr for nbr in G[node]
                          if nbr in removed_nodes] for node in removed_nodes}
    new_components = find_connected_components(removed_adj)

    partitions = [current_partition]
    partitions.extend(new_components)

    logger.info(
        f"Split partition into {len(partitions)} partitions with target cardinality {ci}.")

    return partitions
